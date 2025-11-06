"""
Trading-specific UI Components
Components tailored for trading interfaces with Matrix theming
"""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.tree import Tree
from rich.live import Live
from rich.align import Align
from rich.progress import Progress
from rich.prompt import Prompt, Confirm
from rich.status import Status
from rich.style import Style
from rich.traceback import install
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import asyncio
from decimal import Decimal

from .base_components import BaseComponent, TableComponent, ProgressComponent, InteractiveComponent, AlertComponent
from ..themes.matrix_theme import MatrixTheme


class PortfolioComponent(BaseComponent):
    """Portfolio management UI component"""
    
    def create_portfolio_summary(self, portfolio_data: Dict[str, Any]) -> Panel:
        """Create portfolio summary panel"""
        balance = portfolio_data.get('balance', 0)
        equity = portfolio_data.get('equity', 0)
        buying_power = portfolio_data.get('buying_power', 0)
        day_change = portfolio_data.get('day_change', 0)
        day_change_pct = portfolio_data.get('day_change_pct', 0)
        
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column(style=self.theme.PRIMARY_GREEN, width=15)
        table.add_column(style=self.theme.WHITE)
        
        # Balance
        balance_text = self.format_value(balance, "currency")
        table.add_row("Balance", balance_text)
        
        # Equity
        equity_text = self.format_value(equity, "currency")
        table.add_row("Equity", equity_text)
        
        # Buying Power
        buying_power_text = self.format_value(buying_power, "currency")
        table.add_row("Buying Power", buying_power_text)
        
        # Day Change
        day_change_text = self.format_value(day_change, "currency")
        table.add_row("Day Change", day_change_text)
        
        # Day Change Percentage
        pct_text = self.format_value(day_change_pct, "percentage")
        table.add_row("Day Change %", pct_text)
        
        return Panel(
            table,
            title="[bold green]PORTFOLIO SUMMARY",
            border_style=self.theme.PRIMARY_GREEN,
            padding=(1, 2),
            height=10
        )
    
    def create_position_details(self, position: Dict[str, Any]) -> Panel:
        """Create detailed position view"""
        if not position:
            return Panel(
                Text("No position selected", style=self.theme.DIM_GREEN),
                title="[bold green]POSITION DETAILS",
                border_style=self.theme.PRIMARY_GREEN
            )
        
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column(style=self.theme.PRIMARY_GREEN, width=15)
        table.add_column(style=self.theme.WHITE)
        
        symbol = position.get('symbol', 'N/A')
        side = position.get('side', 'N/A').upper()
        quantity = position.get('quantity', 0)
        avg_entry = position.get('avg_entry_price', 0)
        current_price = position.get('current_price', 0)
        unrealized_pnl = position.get('unrealized_pnl', 0)
        realized_pnl = position.get('realized_pnl', 0)
        margin_used = position.get('margin_used', 0)
        
        table.add_row("Symbol", symbol)
        table.add_row("Side", side)
        table.add_row("Quantity", str(quantity))
        table.add_row("Avg Entry", f"${avg_entry:.2f}")
        table.add_row("Current Price", f"${current_price:.2f}")
        table.add_row("Unrealized P&L", self.format_value(unrealized_pnl, "currency"))
        table.add_row("Realized P&L", self.format_value(realized_pnl, "currency"))
        table.add_row("Margin Used", f"${margin_used:.2f}")
        
        return Panel(
            table,
            title=f"[bold green]POSITION: {symbol}",
            border_style=self.theme.PRIMARY_GREEN,
            padding=(1, 2)
        )


class OrderComponent(BaseComponent):
    """Order management UI component"""
    
    def create_order_entry_form(self) -> Panel:
        """Create order entry form"""
        return Panel(
            Text("Order entry form - Use arrow keys to navigate\nPress Enter to submit order", style=self.theme.MATRIX_CODE),
            title="[bold green]ORDER ENTRY",
            border_style=self.theme.PRIMARY_GREEN,
            padding=(1, 2),
            height=8
        )
    
    def create_order_book(self, bids: List[Dict], asks: List[Dict], max_rows: int = 10) -> Panel:
        """Create order book display"""
        table = Table(show_header=True, box=None, padding=(0, 1))
        table.add_column("Price", style=self.theme.PRIMARY_GREEN, justify="right")
        table.add_column("Size", style=self.theme.DARK_GREEN, justify="right")
        table.add_column("Side", style=self.theme.CYAN, justify="center")
        table.add_column("Size", style=self.theme.DARK_GREEN, justify="right")
        table.add_column("Price", style=self.theme.PRIMARY_GREEN, justify="right")
        
        # Display bids (top of book)
        for i, bid in enumerate(bids[-max_rows:] if len(bids) > max_rows else bids):
            price = f"${bid.get('price', 0):.2f}"
            size = str(bid.get('size', 0))
            
            if i == 0:  # Best bid
                price = f"[{self.theme.SUCCESS}] {price} [/{self.theme.SUCCESS}]"
                size = f"[{self.theme.SUCCESS}] {size} [/{self.theme.SUCCESS}]"
            
            table.add_row(price, size, "BID", "", "")
        
        # Display asks (bottom of book)
        for i, ask in enumerate(asks[:max_rows]):
            price = f"${ask.get('price', 0):.2f}"
            size = str(ask.get('size', 0))
            
            if i == 0:  # Best ask
                price = f"[{self.theme.ERROR}] {price} [/{self.theme.ERROR}]"
                size = f"[{self.theme.ERROR}] {size} [/{self.theme.ERROR}]"
            
            table.add_row("", "", "ASK", size, price)
        
        return Panel(
            table,
            title="[bold green]ORDER BOOK",
            border_style=self.theme.PRIMARY_GREEN,
            padding=(1, 1)
        )
    
    def create_recent_trades(self, trades: List[Dict], max_rows: int = 15) -> Panel:
        """Create recent trades display"""
        if not trades:
            return Panel(
                Text("No recent trades", style=self.theme.DIM_GREEN),
                title="[bold green]RECENT TRADES",
                border_style=self.theme.PRIMARY_GREEN
            )
        
        table = Table(show_header=True, box=None, padding=(0, 1))
        table.add_column("Time", style=self.theme.MATRIX_FADE)
        table.add_column("Price", style=self.theme.PRIMARY_GREEN, justify="right")
        table.add_column("Size", style=self.theme.DARK_GREEN, justify="right")
        table.add_column("Side", style=self.theme.CYAN, justify="center")
        
        for trade in trades[-max_rows:]:
            timestamp = trade.get('timestamp', '')
            price = f"${trade.get('price', 0):.2f}"
            size = str(trade.get('size', 0))
            side = trade.get('side', 'N/A').upper()
            
            # Color price based on trade side
            price_style = self.theme.SUCCESS if side == 'BUY' else self.theme.ERROR
            
            table.add_row(timestamp[:8], f"[{price_style}]{price}[/{price_style}]", size, side)
        
        return Panel(
            table,
            title="[bold green]RECENT TRADES",
            border_style=self.theme.PRIMARY_GREEN,
            padding=(1, 1)
        )


class MarketDataComponent(BaseComponent):
    """Market data display component"""
    
    def create_ticker(self, symbols: List[str]) -> Panel:
        """Create market ticker"""
        ticker_text = Text()
        
        # Create scrolling ticker with symbols and prices
        for symbol in symbols[:10]:  # Limit to 10 symbols
            symbol_text = Text(f" {symbol} ", style=self.theme.PRIMARY_GREEN + " bold")
            price_text = Text(f"${100.00 + hash(symbol) % 50:+.2f} ", style=self.theme.WHITE)
            ticker_text.append(symbol_text)
            ticker_text.append(price_text)
            ticker_text.append(" │ ")
        
        return Panel(
            Align.center(ticker_text),
            title="[bold green]MARKET TICKER",
            border_style=self.theme.PRIMARY_GREEN,
            height=3
        )
    
    def create_market_overview(self, market_data: Dict[str, Any]) -> Panel:
        """Create market overview panel"""
        if not market_data:
            return Panel(
                Text("No market data", style=self.theme.DIM_GREEN),
                title="[bold green]MARKET OVERVIEW",
                border_style=self.theme.PRIMARY_GREEN
            )
        
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column(style=self.theme.PRIMARY_GREEN, width=15)
        table.add_column(style=self.theme.WHITE)
        
        # Add market indices
        indices = market_data.get('indices', {})
        for name, data in indices.items():
            price = f"${data.get('price', 0):.2f}"
            change = data.get('change', 0)
            change_pct = data.get('change_pct', 0)
            
            if change >= 0:
                change_text = f"[{self.theme.SUCCESS}]+${change:.2f} (+{change_pct:.2f}%)[/{self.theme.SUCCESS}]"
            else:
                change_text = f"[{self.theme.ERROR}]${change:.2f} ({change_pct:.2f}%)[/{self.theme.ERROR}]"
            
            table.add_row(name, f"{price} {change_text}")
        
        return Panel(
            table,
            title="[bold green]MARKET OVERVIEW",
            border_style=self.theme.PRIMARY_GREEN,
            padding=(1, 2)
        )
    
    def create_price_chart(self, symbol: str, data: List[Dict]) -> Panel:
        """Create simple price chart display"""
        if not data or len(data) < 2:
            return Panel(
                Text("Insufficient data for chart", style=self.theme.DIM_GREEN),
                title=f"[bold green]PRICE CHART: {symbol}",
                border_style=self.theme.PRIMARY_GREEN,
                height=10
            )
        
        # Simple ASCII chart representation
        prices = [item.get('price', 0) for item in data[-20:]]  # Last 20 data points
        if not prices:
            prices = [100.0] * 10
            
        min_price = min(prices)
        max_price = max(prices)
        price_range = max_price - min_price if max_price > min_price else 1
        
        chart_lines = []
        chart_height = 8
        
        for i in range(chart_height, 0, -1):
            threshold = min_price + (price_range * i / chart_height)
            line = ""
            
            for price in prices:
                if price >= threshold:
                    line += "█"
                else:
                    line += " "
            
            line += f" ${threshold:.2f}"
            chart_lines.append(line)
        
        # Add time labels
        time_label = f"Time: {data[0].get('timestamp', 'N/A')} - {data[-1].get('timestamp', 'N/A')}"
        chart_lines.append(time_label)
        
        chart_text = Text("\n".join(chart_lines))
        chart_text.stylize(self.theme.PRIMARY_GREEN)
        
        return Panel(
            chart_text,
            title=f"[bold green]PRICE CHART: {symbol}",
            border_style=self.theme.PRIMARY_GREEN,
            padding=(1, 2),
            height=12
        )


class RiskComponent(BaseComponent):
    """Risk management UI component"""
    
    def create_risk_dashboard(self, risk_data: Dict[str, Any]) -> Panel:
        """Create comprehensive risk dashboard"""
        if not risk_data:
            return Panel(
                Text("Risk monitoring active", style=self.theme.DIM_GREEN),
                title="[bold green]RISK DASHBOARD",
                border_style=self.theme.PRIMARY_GREEN,
                height=10
            )
        
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column(style=self.theme.PRIMARY_GREEN, width=20)
        table.add_column(style=self.theme.WHITE)
        
        # Risk metrics
        metrics = [
            ("Portfolio Value", f"${risk_data.get('portfolio_value', 0):,.2f}"),
            ("Daily P&L", self.format_value(risk_data.get('daily_pnl', 0), "currency")),
            ("Max Drawdown", f"${risk_data.get('max_drawdown', 0):,.2f}"),
            ("Sharpe Ratio", f"{risk_data.get('sharpe_ratio', 0):.2f}"),
            ("Win Rate", f"{risk_data.get('win_rate', 0):.1f}%"),
            ("Open Positions", str(risk_data.get('open_positions', 0))),
            ("Leverage", f"{risk_data.get('leverage', 1.0):.1f}x"),
            ("Risk Score", str(risk_data.get('risk_score', 'N/A')))
        ]
        
        for metric, value in metrics:
            if isinstance(value, Text):
                table.add_row(metric, value)
            else:
                table.add_row(metric, str(value))
        
        return Panel(
            table,
            title="[bold green]RISK DASHBOARD",
            border_style=self.theme.PRIMARY_GREEN,
            padding=(1, 2),
            height=12
        )
    
    def create_risk_alerts(self, alerts: List[Dict]) -> Panel:
        """Create risk alerts display"""
        if not alerts:
            return Panel(
                Text("No risk alerts", style=self.theme.SUCCESS),
                title="[bold green]RISK ALERTS",
                border_style=self.theme.PRIMARY_GREEN
            )
        
        alert_lines = []
        
        for alert in alerts:
            level = alert.get('level', 'INFO').upper()
            message = alert.get('message', 'No message')
            timestamp = alert.get('timestamp', datetime.now().strftime('%H:%M:%S'))
            
            color_map = {
                'CRITICAL': self.theme.ERROR,
                'HIGH': self.theme.ERROR,
                'MEDIUM': self.theme.WARNING,
                'LOW': self.theme.WARNING,
                'INFO': self.theme.INFO
            }
            
            color = color_map.get(level, self.theme.INFO)
            alert_line = f"[{color}][{timestamp}][/] [{color}]{level}[/{color}]: {message}"
            alert_lines.append(alert_line)
        
        alert_text = Text("\n".join(alert_lines))
        
        return Panel(
            alert_text,
            title="[bold green]RISK ALERTS",
            border_style=self.theme.PRIMARY_GREEN,
            padding=(1, 2)
        )


class StrategyComponent(BaseComponent):
    """Trading strategy UI component"""
    
    def create_strategy_selector(self, strategies: List[Dict]) -> Panel:
        """Create strategy selection interface"""
        if not strategies:
            return Panel(
                Text("No strategies available", style=self.theme.DIM_GREEN),
                title="[bold green]STRATEGY SELECTOR",
                border_style=self.theme.PRIMARY_GREEN
            )
        
        strategy_text = Text()
        
        for i, strategy in enumerate(strategies):
            name = strategy.get('name', f'Strategy {i+1}')
            status = strategy.get('status', 'inactive').upper()
            pnl = strategy.get('total_pnl', 0)
            
            # Create strategy item
            status_text = Text(f"{status}", style=self.theme.PRIMARY_GREEN)
            if status == "ACTIVE":
                status_text.stylize(self.theme.SUCCESS)
            elif status == "PAUSED":
                status_text.stylize(self.theme.WARNING)
            else:
                status_text.stylize(self.theme.DIM_GREEN)
            
            pnl_text = self.format_value(pnl, "currency")
            
            line = Text.assemble(
                f"{i+1:2}. {name:20} ",
                status_text,
                f"   P&L: ",
                pnl_text
            )
            
            strategy_text.append(line)
            strategy_text.append("\n")
        
        return Panel(
            strategy_text,
            title="[bold green]STRATEGY SELECTOR",
            border_style=self.theme.PRIMARY_GREEN,
            padding=(1, 2)
        )
    
    def create_strategy_config(self, strategy: Dict[str, Any]) -> Panel:
        """Create strategy configuration panel"""
        if not strategy:
            return Panel(
                Text("No strategy selected", style=self.theme.DIM_GREEN),
                title="[bold green]STRATEGY CONFIG",
                border_style=self.theme.PRIMARY_GREEN
            )
        
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column(style=self.theme.PRIMARY_GREEN, width=15)
        table.add_column(style=self.theme.WHITE)
        
        config = strategy.get('config', {})
        for key, value in config.items():
            if isinstance(value, (int, float)):
                table.add_row(key.title(), str(value))
            else:
                table.add_row(key.title(), str(value))
        
        return Panel(
            table,
            title=f"[bold green]STRATEGY: {strategy.get('name', 'Unknown')}",
            border_style=self.theme.PRIMARY_GREEN,
            padding=(1, 2)
        )


class BrokerComponent(BaseComponent):
    """Broker connection UI component"""
    
    def create_broker_status(self, brokers: Dict[str, Dict]) -> Panel:
        """Create broker status display"""
        if not brokers:
            return Panel(
                Text("No brokers configured", style=self.theme.DIM_GREEN),
                title="[bold green]BROKER STATUS",
                border_style=self.theme.PRIMARY_GREEN
            )
        
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column(style=self.theme.PRIMARY_GREEN, width=15)
        table.add_column(style=self.theme.WHITE)
        
        for broker_name, broker_data in brokers.items():
            status = broker_data.get('status', 'unknown')
            connected = broker_data.get('connected', False)
            
            if connected:
                status_display = self.theme.create_status_indicator("CONNECTED")
            else:
                status_display = self.theme.create_status_indicator("DISCONNECTED")
            
            table.add_row(f"{broker_name.upper()}", status_display)
        
        return Panel(
            table,
            title="[bold green]BROKER STATUS",
            border_style=self.theme.PRIMARY_GREEN,
            padding=(1, 2),
            height=len(brokers) + 3
        )
    
    def create_connection_log(self, connections: List[Dict], max_logs: int = 10) -> Panel:
        """Create connection log display"""
        if not connections:
            return Panel(
                Text("No connection logs", style=self.theme.DIM_GREEN),
                title="[bold green]CONNECTION LOG",
                border_style=self.theme.PRIMARY_GREEN
            )
        
        log_lines = []
        
        for conn in connections[-max_logs:]:
            timestamp = conn.get('timestamp', datetime.now().strftime('%H:%M:%S'))
            broker = conn.get('broker', 'Unknown')
            action = conn.get('action', 'Unknown')
            status = conn.get('status', 'Unknown')
            
            color = self.theme.SUCCESS if status.lower() == 'success' else self.theme.ERROR
            log_line = f"[{color}][{timestamp}][/] {broker}: {action} ({status})"
            log_lines.append(log_line)
        
        log_text = Text("\n".join(log_lines))
        
        return Panel(
            log_text,
            title="[bold green]CONNECTION LOG",
            border_style=self.theme.PRIMARY_GREEN,
            padding=(1, 2),
            height=max_logs + 3
        )