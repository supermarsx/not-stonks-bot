"""
Matrix Terminal Interface
Main interface combining all components with Matrix theming
"""

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.live import Live
from rich.text import Text
from rich.align import Align
from rich.tree import Tree
from rich.progress import Progress, TaskID
from rich.status import Status
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.screen import Screen
from rich import box
from typing import Dict, List, Optional, Any, Callable
import asyncio
import signal
import sys
from datetime import datetime

from .components.base_components import BaseComponent, PanelComponent, TableComponent, InteractiveComponent, AlertComponent
from .components.trading_components import PortfolioComponent, OrderComponent, MarketDataComponent, RiskComponent, StrategyComponent, BrokerComponent
from .components.realtime_components import RealTimeDataManager, LoadingIndicator
from .components.interactive_elements import (
    FormComponent, OrderEntryForm, BrokerSetupWizard, 
    ConfigurationPanel, AIChatInterface
)
from .themes.matrix_theme import MatrixTheme, MatrixEffects


class MatrixTerminalInterface:
    """
    Main Matrix-themed terminal interface for trading orchestrator
    
    Features:
    - Real-time dashboard with live updates
    - Portfolio and position monitoring
    - Order management interface
    - Market data display
    - Risk monitoring and alerts
    - Strategy selection and configuration
    - Broker connection management
    - AI chat integration
    - Keyboard navigation and shortcuts
    - Matrix visual effects
    """
    
    def __init__(self, console: Console = None):
        self.console = console or Console()
        self.console.set_window_title("Day Trading Orchestrator - Matrix Terminal")
        
        # Initialize components
        self.panel_comp = PanelComponent(self.console)
        self.table_comp = TableComponent(self.console)
        self.interactive_comp = InteractiveComponent(self.console)
        self.alert_comp = AlertComponent(self.console)
        self.portfolio_comp = PortfolioComponent(self.console)
        self.order_comp = OrderComponent(self.console)
        self.market_comp = MarketDataComponent(self.console)
        self.risk_comp = RiskComponent(self.console)
        self.strategy_comp = StrategyComponent(self.console)
        self.broker_comp = BrokerComponent(self.console)
        self.loading_comp = LoadingIndicator(self.console)
        self.realtime_manager = RealTimeDataManager(self.console)
        
        # Initialize interactive components
        self.order_form = OrderEntryForm(self.console)
        self.broker_setup = BrokerSetupWizard(self.console)
        self.config_panel = ConfigurationPanel(self.console)
        self.ai_chat = AIChatInterface(self.console)
        
        # Initialize theme
        self.theme = MatrixTheme()
        
        # Layout setup
        self.layout = Layout()
        self.setup_layout()
        
        # State management
        self.running = False
        self.current_view = "dashboard"
        self.notification_queue = []
        self.alerts = []
        
        # Setup keyboard handlers
        self.setup_key_handlers()
        
        # Mock data for demonstration
        self.mock_data = self.generate_mock_data()
        
    def setup_layout(self):
        """Setup the main layout structure"""
        # Main layout: header, body, footer
        self.layout.split_column(
            Layout(name="header", size=4),
            Layout(name="body"),
            Layout(name="footer", size=3)
        )
        
        # Body: left and right panels
        self.layout["body"].split_row(
            Layout(name="left_panel", ratio=2),
            Layout(name="right_panel", ratio=3)
        )
        
        # Left panel: portfolio and positions
        self.layout["left_panel"].split_column(
            Layout(name="portfolio", size=12),
            Layout(name="positions", ratio=1)
        )
        
        # Right panel: orders, risk, strategy, broker status
        self.layout["right_panel"].split_column(
            Layout(name="orders", size=12),
            Layout(name="risk", size=8),
            Layout(name="bottom_right", ratio=1)
        )
        
        # Bottom right: strategy and broker tabs
        self.layout["bottom_right"].split_row(
            Layout(name="strategy", ratio=1),
            Layout(name="brokers", ratio=1)
        )
        
    def setup_key_handlers(self):
        """Setup keyboard navigation handlers"""
        # This would be enhanced with proper keyboard handling
        self.key_commands = {
            'q': 'quit',
            'r': 'refresh',
            'd': 'dashboard',
            'o': 'orders',
            'p': 'portfolio',
            's': 'settings',
            'h': 'help',
            '?': 'help',
            'f1': 'help',
            'escape': 'dashboard'
        }
    
    def create_header(self) -> Panel:
        """Create the main header with system info"""
        # System status indicators
        status_text = Text()
        
        # Current time
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        time_text = Text(f" {current_time} ", style=self.theme.WHITE + " bold")
        status_text.append(time_text)
        
        # System status
        status_text.append(" │ ")
        status_text.append(self.theme.create_status_indicator("ACTIVE"))
        status_text.append(" │ Matrix Terminal Interface")
        
        # ASCII art header
        header_content = Text()
        header_content.append(MatrixEffects.matrix_rain_effect())
        header_content.append("\n")
        header_content.append(Align.center(status_text))
        
        return Panel(
            header_content,
            style=self.theme.BLACK,
            border_style=self.theme.PRIMARY_GREEN,
            height=4
        )
    
    def create_footer(self) -> Panel:
        """Create the footer with help and shortcuts"""
        help_text = Text()
        
        # Key shortcuts
        shortcuts = [
            ("[b]Q[/b]", "Quit"),
            ("[b]R[/b]", "Refresh"),
            ("[b]D[/b]", "Dashboard"),
            ("[b]O[/b]", "Orders"),
            ("[b]P[/b]", "Portfolio"),
            ("[b]S[/b]", "Settings"),
            ("[b]?[/b]", "Help")
        ]
        
        for key, desc in shortcuts:
            help_text.append(f"{key}: {desc}  ")
        
        return Panel(
            Align.center(help_text),
            style=self.theme.BLACK,
            border_style=self.theme.PRIMARY_GREEN,
            height=3
        )
    
    def generate_mock_data(self) -> Dict[str, Any]:
        """Generate mock data for demonstration"""
        return {
            'portfolio': {
                'balance': 100000,
                'equity': 102500,
                'buying_power': 75000,
                'day_change': 2500,
                'day_change_pct': 2.5
            },
            'positions': [
                {'symbol': 'AAPL', 'side': 'LONG', 'quantity': 100, 'current_price': 150.00, 'unrealized_pnl': 250.00},
                {'symbol': 'TSLA', 'side': 'SHORT', 'quantity': -50, 'current_price': 200.00, 'unrealized_pnl': -125.00},
                {'symbol': 'GOOGL', 'side': 'LONG', 'quantity': 25, 'current_price': 2800.00, 'unrealized_pnl': 450.00}
            ],
            'orders': [
                {'order_id': 'ORD12345', 'symbol': 'AAPL', 'order_type': 'LIMIT', 'side': 'BUY', 'status': 'PENDING'},
                {'order_id': 'ORD12346', 'symbol': 'TSLA', 'order_type': 'MARKET', 'side': 'SELL', 'status': 'FILLED'},
                {'order_id': 'ORD12347', 'symbol': 'GOOGL', 'order_type': 'LIMIT', 'side': 'BUY', 'status': 'PARTIAL'}
            ],
            'risk': {
                'portfolio_value': 102500,
                'daily_pnl': 2500,
                'max_drawdown': -500,
                'sharpe_ratio': 1.8,
                'win_rate': 65.5,
                'open_positions': 3,
                'leverage': 1.3,
                'risk_score': 'MEDIUM'
            },
            'strategies': [
                {'name': 'Trend Follower', 'status': 'active', 'total_pnl': 1250},
                {'name': 'Mean Reversion', 'status': 'paused', 'total_pnl': -200},
                {'name': 'Momentum', 'status': 'active', 'total_pnl': 800}
            ],
            'brokers': {
                'binance': {'status': 'connected', 'connected': True},
                'alpaca': {'status': 'connected', 'connected': True},
                'ibkr': {'status': 'disconnected', 'connected': False}
            }
        }
    
    def render_dashboard(self):
        """Render the main dashboard"""
        try:
            # Update header
            self.layout["header"].update(self.create_header())
            
            # Update left panel
            self.layout["portfolio"].update(
                self.portfolio_comp.create_portfolio_summary(self.mock_data['portfolio'])
            )
            self.layout["positions"].update(
                self.table_comp.create_positions_table(self.mock_data['positions'])
            )
            
            # Update right panel
            self.layout["orders"].update(
                self.table_comp.create_orders_table(self.mock_data['orders'])
            )
            self.layout["risk"].update(
                self.risk_comp.create_risk_dashboard(self.mock_data['risk'])
            )
            self.layout["strategy"].update(
                self.strategy_comp.create_strategy_selector(self.mock_data['strategies'])
            )
            self.layout["brokers"].update(
                self.broker_comp.create_broker_status(self.mock_data['brokers'])
            )
            
            # Update footer
            self.layout["footer"].update(self.create_footer())
            
        except Exception as e:
            self.console.print(f"[red]Error rendering dashboard: {e}[/red]")
    
    async def run_dashboard(self, refresh_rate: float = 1.0):
        """Run the live dashboard"""
        try:
            self.running = True
            
            # Print welcome banner
            self.console.print(self.theme.create_matrix_banner(), style="bold green")
            await asyncio.sleep(1)
            
            # Start real-time data streams
            portfolio_data_source = lambda: self.mock_data['portfolio']
            await self.realtime_manager.start_all_streams(portfolio_data_source)
            
            # Run main dashboard loop
            with Live(
                self.layout,
                console=self.console,
                screen=True,
                refresh_per_second=1/refresh_rate,
                auto_refresh=True
            ):
                try:
                    while self.running:
                        # Render dashboard
                        self.render_dashboard()
                        
                        # Add any new alerts
                        if self.alerts:
                            # Show latest alert in notifications
                            latest_alert = self.alerts[-1]
                            notification_text = f"[{self.theme.WARNING}][ALERT][/] {latest_alert}"
                            self.console.log(notification_text, soft=True)
                        
                        await asyncio.sleep(refresh_rate)
                        
                except KeyboardInterrupt:
                    self.console.print("\n[yellow]Shutting down dashboard...[/yellow]")
                except Exception as e:
                    self.console.print(f"[red]Dashboard error: {e}[/red]")
                finally:
                    await self.shutdown()
                    
        except Exception as e:
            self.console.print(f"[red]Failed to start dashboard: {e}[/red]")
    
    async def run_interactive_mode(self):
        """Run interactive command mode"""
        self.console.print(self.theme.create_matrix_banner(), style="bold green")
        
        commands = {
            'dashboard': self.show_dashboard_menu,
            'portfolio': self.show_portfolio_details,
            'orders': self.show_order_management,
            'strategy': self.show_strategy_selection,
            'brokers': self.show_broker_management,
            'risk': self.show_risk_analysis,
            'settings': self.show_settings,
            'help': self.show_help,
            'quit': self.quit_application,
            'chat': self.show_ai_chat,
            'order': self.show_order_entry,
            'setup': self.show_broker_setup
        }
        
        while True:
            try:
                command = self.interactive_comp.get_user_input(
                    "Matrix Terminal",
                    list(commands.keys()),
                    'dashboard'
                )
                
                if command == 'quit':
                    break
                elif command in commands:
                    await commands[command]()
                else:
                    self.console.print(f"[red]Unknown command: {command}[/red]")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                self.console.print(f"[red]Error: {e}[/red]")
        
        await self.shutdown()
    
    async def show_dashboard_menu(self):
        """Show dashboard information"""
        self.console.clear()
        self.console.print(MatrixEffects.matrix_rain_effect(), style="bold green")
        
        # Show current system status
        system_info = {
            'System Status': 'Active',
            'Uptime': '2h 15m',
            'Active Connections': '3/5',
            'Daily P&L': f"${self.mock_data['portfolio']['day_change']:,.2f}",
            'Open Positions': len(self.mock_data['positions']),
            'Pending Orders': len(self.mock_data['orders'])
        }
        
        self.console.print(
            self.panel_comp.create_info_panel("SYSTEM OVERVIEW", system_info)
        )
        
        input("\nPress Enter to continue...")
    
    async def show_portfolio_details(self):
        """Show detailed portfolio information"""
        self.console.clear()
        
        # Portfolio summary
        self.console.print(
            self.portfolio_comp.create_portfolio_summary(self.mock_data['portfolio'])
        )
        
        # Position details
        self.console.print("\n")
        self.console.print(
            self.panel_comp.create_info_panel("DETAILED POSITIONS", {
                'Total Positions': len(self.mock_data['positions']),
                'Long Positions': len([p for p in self.mock_data['positions'] if p['side'] == 'LONG']),
                'Short Positions': len([p for p in self.mock_data['positions'] if p['side'] == 'SHORT']),
                'Total P&L': f"${sum(p['unrealized_pnl'] for p in self.mock_data['positions']):,.2f}"
            })
        )
        
        input("\nPress Enter to continue...")
    
    async def show_order_management(self):
        """Show order management interface"""
        self.console.clear()
        self.console.print(
            self.panel_comp.create_header_panel("ORDER MANAGEMENT", "Interactive Trading Interface")
        )
        
        # Show current orders
        self.console.print(self.table_comp.create_orders_table(self.mock_data['orders']))
        
        # Order entry form
        self.console.print("\n")
        self.console.print(self.order_comp.create_order_entry_form())
        
        # Show market data for context
        self.console.print("\n")
        symbols = ['AAPL', 'TSLA', 'GOOGL', 'MSFT', 'AMZN']
        self.console.print(self.market_comp.create_ticker(symbols))
        
        input("\nPress Enter to continue...")
    
    async def show_strategy_selection(self):
        """Show strategy selection interface"""
        self.console.clear()
        
        self.console.print(
            self.strategy_comp.create_strategy_selector(self.mock_data['strategies'])
        )
        
        # Strategy configuration
        self.console.print("\n")
        active_strategy = next((s for s in self.mock_data['strategies'] if s['status'] == 'active'), None)
        if active_strategy:
            strategy_config = {
                'Strategy Name': active_strategy['name'],
                'Status': active_strategy['status'],
                'Total P&L': f"${active_strategy['total_pnl']:,.2f}",
                'Parameters': 'Custom configuration'
            }
            self.console.print(
                self.strategy_comp.create_strategy_config(active_strategy)
            )
        
        input("\nPress Enter to continue...")
    
    async def show_broker_management(self):
        """Show broker management interface"""
        self.console.clear()
        
        # Broker status
        self.console.print(
            self.broker_comp.create_broker_status(self.mock_data['brokers'])
        )
        
        # Connection details
        connection_details = {
            'Total Brokers': len(self.mock_data['brokers']),
            'Connected': len([b for b in self.mock_data['brokers'].values() if b['connected']]),
            'Disconnected': len([b for b in self.mock_data['brokers'].values() if not b['connected']]),
            'API Status': 'All systems operational'
        }
        
        self.console.print("\n")
        self.console.print(
            self.panel_comp.create_info_panel("CONNECTION SUMMARY", connection_details)
        )
        
        input("\nPress Enter to continue...")
    
    async def show_risk_analysis(self):
        """Show risk analysis interface"""
        self.console.clear()
        
        # Risk dashboard
        self.console.print(
            self.risk_comp.create_risk_dashboard(self.mock_data['risk'])
        )
        
        # Risk alerts
        self.console.print("\n")
        sample_alerts = [
            {'level': 'MEDIUM', 'message': 'Portfolio exposure above 60%', 'timestamp': datetime.now().strftime('%H:%M:%S')},
            {'level': 'LOW', 'message': 'Daily loss limit 75% utilized', 'timestamp': datetime.now().strftime('%H:%M:%S')}
        ]
        
        self.console.print(
            self.risk_comp.create_risk_alerts(sample_alerts)
        )
        
        input("\nPress Enter to continue...")
    
    async def show_settings(self):
        """Show settings interface"""
        self.console.clear()
        
        settings_options = [
            'Refresh Rate',
            'Color Theme',
            'Alert Thresholds',
            'Broker Configuration',
            'Strategy Parameters',
            'Risk Limits'
        ]
        
        self.console.print(
            self.panel_comp.create_info_panel("SETTINGS", {
                f'{i+1}. {option}': 'Configure...' for i, option in enumerate(settings_options)
            })
        )
        
        selected = self.interactive_comp.get_user_input("Select setting", [str(i+1) for i in range(len(settings_options))])
        setting_index = int(selected) - 1
        
        if 0 <= setting_index < len(settings_options):
            self.console.print(f"\n[{self.theme.INFO}]Configuring {settings_options[setting_index]}[/{self.theme.INFO}]")
            # Add configuration interface here
        
        input("\nPress Enter to continue...")
    
    async def show_help(self):
        """Show help information"""
        self.console.clear()
        
        help_content = {
            'Keyboard Shortcuts': 'Q=Quit, R=Refresh, D=Dashboard, O=Orders, P=Portfolio, S=Settings',
            'Navigation': 'Use arrow keys and Enter to navigate menus',
            'Real-time Updates': 'Dashboard auto-refreshes every 1 second',
            'Matrix Theme': 'Green-on-black terminal interface with ASCII art',
            'Commands': 'Type command names or numbers to execute actions',
            'Alerts': 'Risk alerts and notifications appear in real-time',
            'Support': 'Press Ctrl+C or type quit to exit'
        }
        
        self.console.print(
            self.panel_comp.create_info_panel("HELP & SHORTCUTS", help_content, panel_size=12)
        )
        
        input("\nPress Enter to continue...")
    
    async def show_ai_chat(self):
        """Show AI chat interface"""
        self.console.clear()
        await self.ai_chat.start_chat()
    
    async def show_order_entry(self):
        """Show interactive order entry form"""
        self.console.clear()
        
        order = await self.order_form.get_order_input()
        if order:
            self.console.print(f"\n[{self.theme.SUCCESS}]Order {order['order_id']} submitted successfully![/]")
            self.mock_data['orders'].append(order)
        else:
            self.console.print(f"\n[{self.theme.WARNING}]Order entry cancelled[/]")
        
        input("\nPress Enter to continue...")
    
    async def show_broker_setup(self):
        """Show broker setup wizard"""
        self.console.clear()
        
        brokers = ['alpaca', 'binance', 'ibkr']
        broker_name = self.interactive_comp.get_user_input("Select broker to setup", brokers, 'alpaca')
        
        if broker_name in brokers:
            config = await self.broker_setup.setup_broker(broker_name)
            if config:
                success = await self.broker_setup.test_connection(config)
                if success:
                    self.console.print(f"[{self.theme.SUCCESS}]{broker_name} configured and tested successfully![/]")
                    # Update mock broker status
                    if broker_name in self.mock_data['brokers']:
                        self.mock_data['brokers'][broker_name]['connected'] = True
                        self.mock_data['brokers'][broker_name]['status'] = 'connected'
                else:
                    self.console.print(f"[{self.theme.ERROR}]{broker_name} configuration failed[/]")
            else:
                self.console.print(f"[{self.theme.WARNING}]Broker setup cancelled[/]")
        
        input("\nPress Enter to continue...")
    
    def quit_application(self):
        """Quit the application"""
        self.running = False
        raise SystemExit("Application terminated by user")
    
    async def shutdown(self):
        """Shutdown the interface"""
        try:
            if hasattr(self, 'realtime_manager'):
                await self.realtime_manager.stop_all_streams()
            
            self.console.print(f"\n[{self.theme.WARNING}]Matrix Terminal Interface shutting down...[/]")
            
        except Exception as e:
            self.console.print(f"[red]Error during shutdown: {e}[/red]")
    
    async def run(self, mode: str = "dashboard"):
        """Run the Matrix terminal interface"""
        try:
            # Setup signal handlers for graceful shutdown
            def signal_handler(signum, frame):
                self.console.print(f"\n[{self.theme.WARNING}]Received shutdown signal[/]")
                asyncio.create_task(self.shutdown())
                sys.exit(0)
            
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
            
            if mode == "dashboard":
                await self.run_dashboard()
            elif mode == "interactive":
                await self.run_interactive_mode()
            else:
                raise ValueError(f"Unknown mode: {mode}")
                
        except SystemExit:
            pass  # Normal exit
        except Exception as e:
            self.console.print(f"[red]Fatal error: {e}[/red]")
        finally:
            await self.shutdown()


# Global instance
_matrix_terminal = None

def get_matrix_terminal(console: Console = None) -> MatrixTerminalInterface:
    """Get or create the global Matrix terminal instance"""
    global _matrix_terminal
    if _matrix_terminal is None:
        _matrix_terminal = MatrixTerminalInterface(console)
    return _matrix_terminal

# Example usage and CLI entry point
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Matrix Terminal Interface")
    parser.add_argument("--mode", choices=["dashboard", "interactive"], default="dashboard",
                       help="Interface mode (default: dashboard)")
    parser.add_argument("--refresh-rate", type=float, default=1.0,
                       help="Dashboard refresh rate in seconds (default: 1.0)")
    
    args = parser.parse_args()
    
    async def main():
        terminal = get_matrix_terminal()
        await terminal.run(args.mode)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print(f"Error: {e}")
