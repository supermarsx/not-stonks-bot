"""
Base UI Components
Provides reusable UI components with Matrix theming
"""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.layout import Layout
from rich.tree import Tree
from rich.progress import Progress, TaskID, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.prompt import Prompt, Confirm
from rich.syntax import Syntax
from rich.align import Align
from rich.live import Live
from rich.status import Status
from rich.style import Style
from rich.tree import Tree
from typing import Dict, List, Optional, Any, Union
import asyncio
from datetime import datetime
from ..themes.matrix_theme import MatrixTheme, MatrixEffects


class BaseComponent:
    """Base class for all UI components"""
    
    def __init__(self, console: Console = None):
        self.console = console or Console()
        self.theme = MatrixTheme()
        
    def create_panel(self, content: Any, title: str, border_style: str = "primary") -> Panel:
        """Create a Matrix-styled panel"""
        style_map = {
            "primary": self.theme.PRIMARY_GREEN,
            "secondary": self.theme.DARK_GREEN,
            "accent": self.theme.CYAN,
            "success": self.theme.SUCCESS,
            "warning": self.theme.WARNING,
            "error": self.theme.ERROR
        }
        
        return Panel(
            content,
            title=title,
            border_style=style_map.get(border_style, self.theme.PRIMARY_GREEN),
            title_style=self.theme.get_matrix_styles()["panel_title"],
            padding=(1, 2)
        )
    
    def create_table(self, columns: List[str], show_header: bool = True) -> Table:
        """Create a Matrix-styled table"""
        table = self.theme.create_matrix_table(show_header)
        
        for col in columns:
            table.add_column(col, style=self.theme.PRIMARY_GREEN)
            
        return table
    
    def format_value(self, value: Union[float, int, str], value_type: str = "text") -> Text:
        """Format values with appropriate Matrix styling"""
        if value_type == "currency":
            return self.theme.format_currency(value)
        elif value_type == "percentage":
            return self.theme.format_percentage(value)
        elif value_type == "status":
            return self.theme.create_status_indicator(str(value))
        else:
            text = Text(str(value))
            text.stylize(self.theme.WHITE)
            return text


class PanelComponent(BaseComponent):
    """Matrix-themed panel component"""
    
    def create_header_panel(self, title: str, subtitle: str = "", size: int = 3) -> Panel:
        """Create a header panel"""
        title_text = self.theme.format_matrix_text(title, "header_main")
        if subtitle:
            subtitle_text = self.theme.format_matrix_text(f" - {subtitle}", "header_secondary")
            content = Text.assemble(title_text, subtitle_text)
        else:
            content = title_text
            
        return Panel(
            Align.center(content),
            style=self.theme.BLACK,
            border_style=self.theme.PRIMARY_GREEN,
            height=size
        )
    
    def create_status_panel(self, status: str, details: str = "", panel_size: int = 5) -> Panel:
        """Create a status indicator panel"""
        status_text = self.theme.create_status_indicator(status)
        
        if details:
            details_text = Text(details)
            details_text.stylize(self.theme.DARK_GREEN)
            content = Text.assemble(status_text, "\n", details_text)
        else:
            content = status_text
            
        return Panel(
            content,
            title="[bold green]STATUS",
            border_style=self.theme.PRIMARY_GREEN,
            padding=(1, 1),
            height=panel_size
        )
    
    def create_info_panel(self, title: str, data: Dict[str, Any], panel_size: Optional[int] = None) -> Panel:
        """Create an information panel with key-value pairs"""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column(style=self.theme.PRIMARY_GREEN, width=15)
        table.add_column(style=self.theme.WHITE)
        
        for key, value in data.items():
            if isinstance(value, (int, float)) and key.lower() in ['balance', 'equity', 'pnl', 'profit']:
                value_text = self.format_value(value, "currency")
            elif isinstance(value, str) and key.lower() in ['status', 'state', 'condition']:
                value_text = self.format_value(value, "status")
            else:
                value_text = self.format_value(str(value))
                
            table.add_row(key, value_text)
        
        if panel_size:
            return Panel(
                table,
                title=f"[bold green]{title.upper()}",
                border_style=self.theme.PRIMARY_GREEN,
                padding=(1, 2),
                height=panel_size
            )
        else:
            return Panel(
                table,
                title=f"[bold green]{title.upper()}",
                border_style=self.theme.PRIMARY_GREEN,
                padding=(1, 2)
            )


class TableComponent(BaseComponent):
    """Matrix-themed table component"""
    
    def create_data_table(self, data: List[Dict], key_mapping: Dict[str, str] = None, max_rows: int = 10) -> Table:
        """Create a data table with automatic styling"""
        if not data:
            table = self.create_table(["No Data"])
            table.add_row("No data available")
            return table
        
        # Extract columns from first row
        columns = list(data[0].keys())
        if key_mapping:
            columns = [key_mapping.get(col, col) for col in columns]
        
        table = self.create_table(columns)
        
        # Add rows with formatting
        for row in data[:max_rows]:
            formatted_row = []
            for col in columns:
                original_col = col
                if key_mapping:
                    # Find original key
                    original_col = next((k for k, v in key_mapping.items() if v == col), col)
                
                value = row.get(original_col, "N/A")
                
                # Auto-format based on column type
                col_lower = col.lower()
                if 'pnl' in col_lower or 'profit' in col_lower or 'loss' in col_lower:
                    formatted_value = self.format_value(float(value), "currency")
                elif 'percent' in col_lower or '%' in str(value):
                    formatted_value = self.format_value(float(value), "percentage")
                elif 'status' in col_lower or 'state' in col_lower:
                    formatted_value = self.format_value(str(value), "status")
                else:
                    formatted_value = self.format_value(str(value))
                
                formatted_row.append(formatted_value)
            
            table.add_row(*formatted_row)
        
        return table
    
    def create_positions_table(self, positions: List[Dict], max_rows: int = 8) -> Panel:
        """Create a specialized positions table"""
        columns = ["Symbol", "Side", "Quantity", "Price", "P&L"]
        
        if not positions:
            table = self.create_table(columns)
            table.add_row("No positions", "-", "-", "-", "-")
            return self.create_panel(table, "POSITIONS", "primary")
        
        table = self.create_table(columns)
        
        for pos in positions[:max_rows]:
            symbol = pos.get('symbol', 'N/A')
            side = pos.get('side', 'N/A').upper()
            qty = str(pos.get('quantity', 0))
            price = f"${pos.get('current_price', 0):.2f}"
            pnl = pos.get('unrealized_pnl', 0)
            pnl_text = self.format_value(pnl, "currency")
            
            table.add_row(symbol, side, qty, price, pnl_text)
        
        return self.create_panel(table, "POSITIONS", "primary")
    
    def create_orders_table(self, orders: List[Dict], max_rows: int = 8) -> Panel:
        """Create a specialized orders table"""
        columns = ["ID", "Symbol", "Type", "Side", "Status"]
        
        if not orders:
            table = self.create_table(columns)
            table.add_row("No orders", "-", "-", "-", "-")
            return self.create_panel(table, "ORDERS", "primary")
        
        table = self.create_table(columns)
        
        for order in orders[:max_rows]:
            order_id = str(order.get('order_id', 'N/A'))[:8]
            symbol = order.get('symbol', 'N/A')
            order_type = order.get('order_type', 'N/A').upper()
            side = order.get('side', 'N/A').upper()
            status = order.get('status', 'N/A').upper()
            
            status_text = self.format_value(status, "status")
            
            table.add_row(order_id, symbol, order_type, side, status_text)
        
        return self.create_panel(table, "ORDERS", "primary")
    
    def create_risk_table(self, risk_data: Dict[str, Any]) -> Panel:
        """Create a risk metrics table"""
        if not risk_data:
            return self.create_info_panel("RISK METRICS", {"Status": "No risk data available"})
        
        risk_display = {
            "Daily P&L": risk_data.get('daily_pnl', 0),
            "Max Drawdown": risk_data.get('max_drawdown', 0),
            "Open Orders": risk_data.get('open_orders', 0),
            "Exposure %": risk_data.get('exposure_pct', 0),
            "Risk Score": risk_data.get('risk_score', 'N/A')
        }
        
        return self.create_info_panel("RISK METRICS", risk_display, panel_size=8)


class ProgressComponent(BaseComponent):
    """Matrix-themed progress component"""
    
    def create_progress(self, description: str = "Processing...") -> Progress:
        """Create a Matrix-styled progress bar"""
        return Progress(
            TextColumn(
                f"[{self.theme.PRIMARY_GREEN}]{description}[/{self.theme.PRIMARY_GREEN}]",
                style=self.theme.PRIMARY_GREEN
            ),
            BarColumn(
                bar_style=self.theme.PRIMARY_GREEN,
                complete_style=self.theme.SUCCESS,
                finished_style=self.theme.SUCCESS
            ),
            TextColumn(
                "[progress.percentage]{task.percentage:>3.0f}%",
                style=self.theme.DARK_GREEN
            ),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            expand=True,
            console=self.console
        )
    
    def create_loading_status(self, message: str = "Loading...") -> Status:
        """Create a Matrix-styled loading status"""
        loading_text = MatrixEffects.create_loading_text(message)
        return Status(
            loading_text,
            console=self.console,
            style=self.theme.MATRIX_CODE
        )


class InteractiveComponent(BaseComponent):
    """Matrix-themed interactive components"""
    
    def get_user_input(self, prompt_text: str, choices: List[str] = None, default: str = None) -> str:
        """Get user input with Matrix styling"""
        styled_prompt = f"[{self.theme.PRIMARY_GREEN}]{prompt_text}[/{self.theme.PRIMARY_GREEN}]"
        
        if choices:
            styled_prompt += f" [choices: {', '.join(choices)}]"
            if default:
                styled_prompt += f" [default: {default}]"
            
            return Prompt.ask(
                styled_prompt, 
                choices=choices,
                default=default or choices[0]
            )
        else:
            return Prompt.ask(styled_prompt)
    
    def get_confirmation(self, message: str, default: bool = False) -> bool:
        """Get yes/no confirmation with Matrix styling"""
        styled_message = f"[{self.theme.PRIMARY_GREEN}]{message}[/{self.theme.PRIMARY_GREEN}]"
        return Confirm.ask(styled_message, default=default)
    
    def create_tree(self, title: str = "System Tree", data: Dict[str, Any] = None) -> Tree:
        """Create a Matrix-styled tree"""
        tree = self.theme.create_matrix_tree(title)
        
        if data:
            self._add_tree_nodes(tree, data)
        
        return tree
    
    def _add_tree_nodes(self, tree: Tree, data: Dict[str, Any], parent: str = ""):
        """Recursively add nodes to tree"""
        for key, value in data.items():
            node_key = f"{parent}.{key}" if parent else key
            
            if isinstance(value, dict):
                branch = tree.add(f"[{self.theme.PRIMARY_GREEN}]{key}[/{self.theme.PRIMARY_GREEN}]")
                self._add_tree_nodes(branch, value, node_key)
            elif isinstance(value, list):
                branch = tree.add(f"[{self.theme.PRIMARY_GREEN}]{key}[/{self.theme.PRIMARY_GREEN}] ({len(value)} items)")
                for i, item in enumerate(value[:5]):  # Limit to first 5 items
                    if isinstance(item, dict):
                        branch.add(f"[{self.theme.CYAN}][{i+1}][/{self.theme.CYAN}] {list(item.keys())[0] if item else 'Empty'}")
                    else:
                        branch.add(f"[{self.theme.CYAN}][{i+1}][/{self.theme.CYAN}] {str(item)}")
            else:
                display_value = str(value)
                if len(display_value) > 30:
                    display_value = display_value[:27] + "..."
                
                tree.add(f"[{self.theme.DARK_GREEN}]{key}[/{self.theme.DARK_GREEN}]: [{self.theme.WHITE}]{display_value}[/{self.theme.WHITE}]")


class AlertComponent(BaseComponent):
    """Matrix-themed alert and notification components"""
    
    def create_alert(self, message: str, alert_type: str = "info", timeout: int = 5) -> Panel:
        """Create an alert message"""
        style_map = {
            "success": ("SUCCESS", self.theme.SUCCESS),
            "warning": ("WARNING", self.theme.WARNING),
            "error": ("ERROR", self.theme.ERROR),
            "info": ("INFO", self.theme.INFO)
        }
        
        title, color = style_map.get(alert_type, style_map["info"])
        
        alert_text = Text(message)
        alert_text.stylize(color)
        
        return Panel(
            alert_text,
            title=f"[bold {color}]{title}[/bold {color}]",
            border_style=color,
            padding=(1, 2)
        )
    
    def create_notification_center(self, notifications: List[Dict], max_notifications: int = 5) -> Panel:
        """Create a notification center panel"""
        if not notifications:
            return self.create_panel(
                Text("No notifications", style=self.theme.DIM_GREEN),
                "NOTIFICATIONS"
            )
        
        content_lines = []
        
        for notif in notifications[:max_notifications]:
            timestamp = notif.get('timestamp', datetime.now().strftime('%H:%M:%S'))
            message = notif.get('message', 'No message')
            notif_type = notif.get('type', 'info')
            
            type_color = {
                'success': self.theme.SUCCESS,
                'warning': self.theme.WARNING,
                'error': self.theme.ERROR,
                'info': self.theme.INFO
            }.get(notif_type, self.theme.INFO)
            
            line = f"[{type_color}][{timestamp}][/] {message}"
            content_lines.append(line)
        
        content = Text("\n".join(content_lines))
        
        return Panel(
            content,
            title="[bold green]NOTIFICATIONS",
            border_style=self.theme.PRIMARY_GREEN,
            padding=(1, 2)
        )