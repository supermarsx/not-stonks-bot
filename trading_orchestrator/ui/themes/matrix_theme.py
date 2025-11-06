"""
Matrix Theme Configuration
Defines the Matrix color palette and styling for the terminal interface
"""

from rich.style import Style
from rich.color import Color
from rich.theme import Theme
from rich.text import Text
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, TaskID
from rich.prompt import Prompt
from rich.tree import Tree
from rich.live import Live
from rich.layout import Layout
from rich.syntax import Syntax
from typing import Dict, Any
import os


class MatrixTheme:
    """
    Matrix-themed styling configuration for the terminal interface
    """
    
    # Matrix Color Palette
    PRIMARY_GREEN = "bright_green"
    DARK_GREEN = "green"
    DIM_GREEN = "dim_green" 
    CYAN = "cyan"
    WHITE = "white"
    BLACK = "black"
    RED = "red"
    YELLOW = "yellow"
    
    # Extended Colors for Matrix effects
    MATRIX_CODE = "bright_cyan"
    MATRIX_FADE = "dim_cyan"
    SUCCESS = "bright_green"
    WARNING = "bright_yellow"
    ERROR = "bright_red"
    INFO = "cyan"
    
    # ASCII Art Elements
    ASCII_HEADER = """
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║  ████████╗ ██████╗  █████╗ ██████╗ ██╗      █████╗  ██████╗  ██████╗ ███████╗ ║
║  ╚══██╔══╝██╔════╝ ██╔══██╗██╔══██╗██║     ██╔══██╗██╔═══██╗██╔════╝ ██╔════╝ ║
║     ██║   ██║  ███╗███████║██████╔╝██║     ███████║██║   ██║██║  ███╗█████╗   ║
║     ██║   ██║   ██║██╔══██║██╔══██╗██║     ██╔══██║██║   ██║██║   ██║██╔══╝   ║
║     ██║   ╚██████╔╝██║  ██║██║  ██║███████╗██║  ██║╚██████╔╝╚██████╔╝███████╗ ║
║     ╚═╝    ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝ ╚═════╝  ╚═════╝ ╚══════╝ ║
║                                                                               ║
║                    DAY TRADING ORCHESTRATOR SYSTEM                           ║
║                        Matrix Terminal Interface                             ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
    """
    
    ASCII_DIVIDER = "═" * 80
    
    @classmethod
    def get_theme_colors(cls) -> Dict[str, str]:
        """Get the Matrix color theme"""
        return {
            # Primary Matrix colors
            "primary": cls.PRIMARY_GREEN,
            "secondary": cls.DARK_GREEN,
            "accent": cls.CYAN,
            "text": cls.WHITE,
            "background": cls.BLACK,
            
            # Status colors
            "success": cls.SUCCESS,
            "warning": cls.WARNING,
            "error": cls.ERROR,
            "info": cls.INFO,
            
            # Matrix-specific colors
            "matrix_code": cls.MATRIX_CODE,
            "matrix_fade": cls.MATRIX_FADE,
            "ascii_art": cls.PRIMARY_GREEN
        }
    
    @classmethod
    def get_rich_theme(cls) -> Theme:
        """Get Rich theme configuration"""
        colors = cls.get_theme_colors()
        return Theme({
            # Base colors
            "primary": colors["primary"],
            "secondary": colors["secondary"],
            "accent": colors["accent"],
            
            # Status colors
            "success": colors["success"],
            "warning": colors["warning"],
            "error": colors["error"],
            "info": colors["info"],
            
            # Matrix-specific
            "matrix": colors["matrix_code"],
            "ascii": colors["ascii_art"],
            
            # Text styles
            "bold": "bold",
            "italic": "italic",
            "dim": "dim",
            "underline": "underline",
        })
    
    @classmethod
    def get_matrix_styles(cls) -> Dict[str, Style]:
        """Get Matrix-specific text styles"""
        return {
            # Header styles
            "header_main": Style(color=cls.PRIMARY_GREEN, bold=True, underline=True),
            "header_secondary": Style(color=cls.DARK_GREEN, bold=True),
            
            # Panel styles
            "panel_border": Style(color=cls.PRIMARY_GREEN),
            "panel_title": Style(color=cls.PRIMARY_GREEN, bold=True),
            
            # Text styles
            "success": Style(color=cls.SUCCESS, bold=True),
            "warning": Style(color=cls.WARNING, bold=True),
            "error": Style(color=cls.ERROR, bold=True),
            "info": Style(color=cls.INFO),
            "dim": Style(color=cls.MATRIX_FADE),
            
            # Matrix code effects
            "matrix_code": Style(color=cls.MATRIX_CODE, bold=False),
            "matrix_typewriter": Style(color=cls.DARK_GREEN, italic=True),
            
            # Interactive elements
            "button": Style(color=cls.PRIMARY_GREEN, bold=True, bgcolor="black"),
            "button_hover": Style(color=cls.CYAN, bold=True, bgcolor="black"),
            "input": Style(color=cls.WHITE, bgcolor="black", bold=False),
            
            # Data display
            "positive": Style(color=cls.SUCCESS, bold=True),
            "negative": Style(color=cls.ERROR, bold=True),
            "neutral": Style(color=cls.DARK_GREEN),
            "highlight": Style(color=cls.CYAN, bold=True),
        }
    
    @classmethod
    def create_matrix_banner(cls, title: str = "DAY TRADING SYSTEM") -> str:
        """Create Matrix-style ASCII banner"""
        return f"""
{cls.ASCII_HEADER}

SYSTEM: {title}
VERSION: v1.0 Matrix Terminal Interface
STATUS: Active & Monitoring

{cls.ASCII_DIVIDER}
        """.strip()
    
    @classmethod
    def create_matrix_divider(cls, char: str = "─", length: int = 60) -> str:
        """Create Matrix-style divider"""
        return char * length
    
    @classmethod
    def format_matrix_text(cls, text: str, style: str = "primary") -> Text:
        """Format text with Matrix styling"""
        matrix_styles = cls.get_matrix_styles()
        style_obj = matrix_styles.get(style, matrix_styles["info"])
        
        text_obj = Text(text)
        text_obj.stylize(style_obj)
        return text_obj
    
    @classmethod
    def create_matrix_panel(cls, content: Any, title: str, style: str = "panel_title") -> Panel:
        """Create Matrix-styled panel"""
        title_style = cls.get_matrix_styles()[style]
        
        return Panel(
            content,
            title=title,
            border_style=cls.PRIMARY_GREEN,
            title_style=title_style,
            padding=(1, 2),
            expand=False
        )
    
    @classmethod
    def create_matrix_table(cls, show_header: bool = True, box_style: str = "rounded") -> Table:
        """Create Matrix-styled table"""
        table = Table(
            show_header=show_header,
            box=box_style,
            header_style=cls.PRIMARY_GREEN,
            show_lines=False,
            expand=True,
            pad_edge=False,
            padding=(0, 1),
            row_styles=[Style(), Style(color="dim")]
        )
        return table
    
    @classmethod
    def get_matrix_progress_styles(cls) -> Dict[str, str]:
        """Get Matrix-styled progress bar colors"""
        return {
            "description": cls.PRIMARY_GREEN,
            "progress": cls.CYAN,
            "total": cls.DARK_GREEN,
            "bar": cls.PRIMARY_GREEN,
            "complete": cls.SUCCESS,
        }
    
    @classmethod
    def format_currency(cls, amount: float, color_negative: bool = True) -> Text:
        """Format currency with Matrix styling"""
        text = Text(f"${amount:,.2f}")
        
        if amount >= 0:
            if color_negative:
                text.stylize(cls.SUCCESS)
        else:
            text.stylize(cls.ERROR)
            
        return text
    
    @classmethod 
    def format_percentage(cls, value: float, color_negative: bool = True) -> Text:
        """Format percentage with Matrix styling"""
        text = Text(f"{value:.2f}%")
        
        if value >= 0:
            if color_negative:
                text.stylize(cls.SUCCESS)
        else:
            text.stylize(cls.ERROR)
            
        return text
    
    @classmethod
    def create_status_indicator(cls, status: str) -> Text:
        """Create Matrix-style status indicator"""
        status_colors = {
            "connected": cls.SUCCESS,
            "active": cls.SUCCESS,
            "running": cls.SUCCESS,
            "pending": cls.WARNING,
            "error": cls.ERROR,
            "failed": cls.ERROR,
            "inactive": cls.DARK_GREEN,
            "unknown": cls.MATRIX_FADE
        }
        
        status_upper = status.upper()
        color = status_colors.get(status.lower(), cls.INFO)
        
        text = Text(f"● {status_upper}")
        text.stylize(color)
        return text
    
    @classmethod
    def create_matrix_tree(cls, title: str = "System Tree") -> Tree:
        """Create Matrix-styled tree"""
        tree_style = Style(color=cls.PRIMARY_GREEN, bold=True)
        
        tree = Tree(
            f"[{cls.PRIMARY_GREEN} bold]{title}[/{cls.PRIMARY_GREEN} bold]",
            style=cls.PRIMARY_GREEN,
            guide_style=cls.DARK_GREEN
        )
        return tree


# Matrix Animation Effects
class MatrixEffects:
    """
    Matrix-style animation and visual effects
    """
    
    @staticmethod
    def matrix_rain_effect() -> str:
        """Create matrix rain effect"""
        return """
        ████████╗ █████╗  █████╗ ██████╗ ██╗      █████╗ 
        ╚══██╔══╝██╔══██╗██╔══██╗██╔══██╗██║     ██╔══██╗
           ██║   ███████║███████║██████╔╝██║     ███████║
           ██║   ██╔══██║██╔══██║██╔══██╗██║     ██╔══██║
           ██║   ██║  ██║██║  ██║██║  ██║███████╗██║  ██║
           ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝
        """
    
    @staticmethod
    def create_loading_text(text: str = "LOADING") -> str:
        """Create Matrix-style loading text"""
        matrix_chars = "▓▓░░░▓░▓▓▓░░▓░░▓▓▓▓░░"
        return f"{text} {matrix_chars}"
    
    @staticmethod
    def matrix_border(char: str = "═", length: int = 80) -> str:
        """Create Matrix border"""
        return char * length
    
    @staticmethod 
    def digital_clock() -> str:
        """Matrix-style digital clock"""
        from datetime import datetime
        return datetime.now().strftime("║ %H:%M:%S ║")


# Global theme instance
matrix_theme = MatrixTheme()