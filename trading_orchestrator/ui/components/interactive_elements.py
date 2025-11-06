"""
Interactive Elements
Forms, configuration panels, and setup wizards for the Matrix terminal interface
"""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.align import Align
from rich.prompt import Prompt, Confirm, IntPrompt, FloatPrompt
from rich.syntax import Syntax
from rich.progress import Progress, SpinnerColumn, TextColumn, TaskID
from rich.status import Status
from rich.tree import Tree
from rich.screen import Screen
from typing import Dict, List, Optional, Any, Union, Callable
import asyncio
from datetime import datetime
import re

from .base_components import BaseComponent, InteractiveComponent
from .trading_components import BrokerComponent
from ..themes.matrix_theme import MatrixTheme


class FormComponent(BaseComponent):
    """Matrix-themed form components"""
    
    def __init__(self, console: Console = None):
        super().__init__(console)
        self.form_fields = []
        self.field_validators = {}
        self.field_types = {}
        
    def add_field(self, name: str, label: str, field_type: str = "text", required: bool = True, 
                  default: Any = None, choices: List[str] = None, validator: Callable = None):
        """Add a field to the form"""
        field = {
            'name': name,
            'label': label,
            'type': field_type,
            'required': required,
            'default': default,
            'choices': choices,
            'validator': validator
        }
        
        self.form_fields.append(field)
        if validator:
            self.field_validators[name] = validator
        if field_type:
            self.field_types[name] = field_type
    
    def create_input_prompt(self, field: Dict) -> Any:
        """Create input prompt for a field"""
        label = field['label']
        field_type = field['type']
        choices = field.get('choices')
        default = field.get('default')
        
        if choices:
            return Prompt.ask(
                f"[{self.theme.PRIMARY_GREEN}]{label}[/{self.theme.PRIMARY_GREEN}]",
                choices=choices,
                default=default
            )
        elif field_type == "integer":
            return IntPrompt.ask(f"[{self.theme.PRIMARY_GREEN}]{label}[/{self.theme.PRIMARY_GREEN}]", default=default)
        elif field_type == "float":
            return FloatPrompt.ask(f"[{self.theme.PRIMARY_GREEN}]{label}[/{self.theme.PRIMARY_GREEN}]", default=default)
        elif field_type == "password":
            from getpass import getpass
            return getpass(f"[{self.theme.PRIMARY_GREEN}]{label}[/{self.theme.PRIMARY_GREEN}]")
        elif field_type == "boolean":
            return Confirm.ask(f"[{self.theme.PRIMARY_GREEN}]{label}[/{self.theme.PRIMARY_GREEN}]", default=default)
        else:
            return Prompt.ask(f"[{self.theme.PRIMARY_GREEN}]{label}[/{self.theme.PRIMARY_GREEN}]", default=default)
    
    def validate_field(self, name: str, value: Any) -> bool:
        """Validate a field value"""
        if name in self.field_validators:
            try:
                return self.field_validators[name](value)
            except Exception:
                return False
        return True
    
    async def display_form(self, title: str = "Form") -> Dict[str, Any]:
        """Display and process a form"""
        self.console.print("\n")
        
        # Create form header
        header_text = Text()
        header_text.append(f"{title}\n", style=self.theme.header_main)
        header_text.append(self.theme.create_matrix_divider())
        
        self.console.print(Panel(header_text, border_style=self.theme.PRIMARY_GREEN))
        
        results = {}
        
        for field in self.form_fields:
            try:
                value = self.create_input_prompt(field)
                
                # Validate if required
                if field['required'] and not value and value != 0:
                    self.console.print(f"[{self.theme.ERROR}]{field['label']} is required[/]")
                    continue
                
                # Validate if validator exists
                if not self.validate_field(field['name'], value):
                    self.console.print(f"[{self.theme.ERROR}]Invalid {field['label']}[/]")
                    continue
                
                results[field['name']] = value
                
                # Show field value
                value_display = str(value)
                if len(value_display) > 50:
                    value_display = value_display[:47] + "..."
                
                self.console.print(f"[{self.theme.SUCCESS}]✓ {field['label']}: {value_display}[/]")
                
            except KeyboardInterrupt:
                self.console.print(f"\n[{self.theme.WARNING}]Form cancelled[/]")
                return {}
            except Exception as e:
                self.console.print(f"[{self.theme.ERROR}]Error processing {field['label']}: {e}[/]")
        
        # Form summary
        if results:
            self.console.print(f"\n[{self.theme.SUCCESS}]Form completed successfully[/]")
        
        return results


class OrderEntryForm:
    """Interactive order entry form"""
    
    def __init__(self, console: Console = None):
        self.console = console or Console()
        self.theme = MatrixTheme()
        
    async def get_order_input(self) -> Dict[str, Any]:
        """Get order input from user"""
        self.console.print(f"\n[{self.theme.PRIMARY_GREEN}]=== ORDER ENTRY FORM ===[/]\n", style="bold green")
        
        # Symbol selection
        symbols = ['AAPL', 'TSLA', 'GOOGL', 'MSFT', 'AMZN', 'META', 'NVDA', 'AMD', 'INTC', 'NFLX']
        symbol = Prompt.ask(
            f"[{self.theme.PRIMARY_GREEN}]Symbol[/]",
            choices=symbols,
            default='AAPL'
        )
        
        # Order type
        order_types = ['MARKET', 'LIMIT', 'STOP', 'STOP_LIMIT']
        order_type = Prompt.ask(
            f"[{self.theme.PRIMARY_GREEN}]Order Type[/]",
            choices=order_types,
            default='MARKET'
        )
        
        # Side
        sides = ['BUY', 'SELL']
        side = Prompt.ask(
            f"[{self.theme.PRIMARY_GREEN}]Side[/]",
            choices=sides,
            default='BUY'
        )
        
        # Quantity
        quantity = IntPrompt.ask(
            f"[{self.theme.PRIMARY_GREEN}]Quantity[/]",
            default=100,
            min=1
        )
        
        # Price (for limit orders)
        price = None
        if order_type in ['LIMIT', 'STOP_LIMIT']:
            price = FloatPrompt.ask(
                f"[{self.theme.PRIMARY_GREEN}]Price[/]",
                default=100.00,
                min=0.01
            )
        
        # Stop price (for stop orders)
        stop_price = None
        if order_type in ['STOP', 'STOP_LIMIT']:
            stop_price = FloatPrompt.ask(
                f"[{self.theme.PRIMARY_GREEN}]Stop Price[/]",
                default=90.00,
                min=0.01
            )
        
        # Time in force
        tif_options = ['DAY', 'GTC', 'IOC', 'FOK']
        time_in_force = Prompt.ask(
            f"[{self.theme.PRIMARY_GREEN}]Time in Force[/]",
            choices=tif_options,
            default='DAY'
        )
        
        # Review order
        self.console.print(f"\n[{self.theme.PRIMARY_GREEN}]=== ORDER REVIEW ===[/]\n", style="bold green")
        
        review_table = Table(show_header=False, box=None)
        review_table.add_column(style=self.theme.PRIMARY_GREEN, width=15)
        review_table.add_column(style=self.theme.WHITE)
        
        review_table.add_row("Symbol:", symbol)
        review_table.add_row("Side:", side)
        review_table.add_row("Order Type:", order_type)
        review_table.add_row("Quantity:", str(quantity))
        if price:
            review_table.add_row("Price:", f"${price:.2f}")
        if stop_price:
            review_table.add_row("Stop Price:", f"${stop_price:.2f}")
        review_table.add_row("Time in Force:", time_in_force)
        
        self.console.print(review_table)
        
        # Confirm order
        confirm = Confirm.ask(f"\n[{self.theme.WARNING}]Submit this order?[/]")
        
        if not confirm:
            self.console.print(f"[{self.theme.WARNING}]Order cancelled[/]")
            return {}
        
        # Simulate order submission
        with Status(f"[{self.theme.MATRIX_CODE}]Submitting order...[/]", console=self.console):
            await asyncio.sleep(2)  # Simulate network delay
        
        order_id = f"ORD{int(datetime.now().timestamp())}"
        
        return {
            'order_id': order_id,
            'symbol': symbol,
            'side': side,
            'order_type': order_type,
            'quantity': quantity,
            'price': price,
            'stop_price': stop_price,
            'time_in_force': time_in_force,
            'status': 'PENDING',
            'timestamp': datetime.now()
        }


class BrokerSetupWizard:
    """Broker configuration setup wizard"""
    
    def __init__(self, console: Console = None):
        self.console = console or Console()
        self.theme = MatrixTheme()
        
    async def setup_broker(self, broker_name: str) -> Dict[str, Any]:
        """Setup broker configuration"""
        self.console.print(f"\n[{self.theme.PRIMARY_GREEN}]=== {broker_name.upper()} SETUP WIZARD ===[/]\n", style="bold green")
        
        if broker_name.lower() == 'alpaca':
            return await self._setup_alpaca()
        elif broker_name.lower() == 'binance':
            return await self._setup_binance()
        elif broker_name.lower() == 'ibkr':
            return await self._setup_ibkr()
        else:
            self.console.print(f"[{self.theme.ERROR}]Unknown broker: {broker_name}[/]")
            return {}
    
    async def _setup_alpaca(self) -> Dict[str, Any]:
        """Setup Alpaca configuration"""
        config = {}
        
        self.console.print(f"[{self.theme.INFO}]Alpaca API Setup[/]\n")
        
        # API Key
        api_key = Prompt.ask(f"[{self.theme.PRIMARY_GREEN}]API Key[/]", password=True)
        if not api_key:
            self.console.print(f"[{self.theme.ERROR}]API Key is required[/]")
            return {}
        
        # Secret Key
        secret_key = Prompt.ask(f"[{self.theme.PRIMARY_GREEN}]Secret Key[/]", password=True)
        if not secret_key:
            self.console.print(f"[{self.theme.ERROR}]Secret Key is required[/]")
            return {}
        
        # Paper trading
        paper_trading = Confirm.ask(f"[{self.theme.PRIMARY_GREEN}]Use Paper Trading?[/]", default=True)
        
        # Environment
        if paper_trading:
            base_url = "https://paper-api.alpaca.markets"
        else:
            base_url = "https://api.alpaca.markets"
        
        return {
            'name': 'alpaca',
            'api_key': api_key,
            'secret_key': secret_key,
            'base_url': base_url,
            'paper_trading': paper_trading,
            'configured': True
        }
    
    async def _setup_binance(self) -> Dict[str, Any]:
        """Setup Binance configuration"""
        config = {}
        
        self.console.print(f"[{self.theme.INFO}]Binance API Setup[/]\n")
        
        # API Key
        api_key = Prompt.ask(f"[{self.theme.PRIMARY_GREEN}]API Key[/]", password=True)
        if not api_key:
            self.console.print(f"[{self.theme.ERROR}]API Key is required[/]")
            return {}
        
        # Secret Key
        secret_key = Prompt.ask(f"[{self.theme.PRIMARY_GREEN}]Secret Key[/]", password=True)
        if not secret_key:
            self.console.print(f"[{self.theme.ERROR}]Secret Key is required[/]")
            return {}
        
        # Testnet
        testnet = Confirm.ask(f"[{self.theme.PRIMARY_GREEN}]Use Testnet?[/]", default=True)
        
        # Environment
        if testnet:
            base_url = "https://testnet.binance.vision"
        else:
            base_url = "https://api.binance.com"
        
        return {
            'name': 'binance',
            'api_key': api_key,
            'secret_key': secret_key,
            'base_url': base_url,
            'testnet': testnet,
            'configured': True
        }
    
    async def _setup_ibkr(self) -> Dict[str, Any]:
        """Setup Interactive Brokers configuration"""
        config = {}
        
        self.console.print(f"[{self.theme.INFO}]Interactive Brokers TWS Setup[/]\n")
        
        # Host
        host = Prompt.ask(f"[{self.theme.PRIMARY_GREEN}]TWS Host[/]", default='127.0.0.1')
        
        # Port
        port = IntPrompt.ask(f"[{self.theme.PRIMARY_GREEN}]TWS Port[/]", default=7497, min=1, max=65535)
        
        # Client ID
        client_id = IntPrompt.ask(f"[{self.theme.PRIMARY_GREEN}]Client ID[/]", default=1, min=1)
        
        # Connection timeout
        timeout = IntPrompt.ask(f"[{self.theme.PRIMARY_GREEN}]Connection Timeout (seconds)[/]", default=30, min=5, max=300)
        
        return {
            'name': 'ibkr',
            'host': host,
            'port': port,
            'client_id': client_id,
            'timeout': timeout,
            'configured': True
        }
    
    async def test_connection(self, config: Dict[str, Any]) -> bool:
        """Test broker connection"""
        broker_name = config.get('name', 'unknown')
        
        self.console.print(f"\n[{self.theme.PRIMARY_GREEN}]Testing {broker_name} connection...[/]\n")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            task = progress.add_task("Connecting...", total=None)
            
            # Simulate connection test
            await asyncio.sleep(3)
            
            # Mock connection result
            import random
            if random.random() > 0.3:  # 70% success rate for demo
                progress.update(task, description="Connection successful!")
                self.console.print(f"[{self.theme.SUCCESS}]✓ {broker_name} connected successfully[/]")
                return True
            else:
                progress.update(task, description="Connection failed!")
                self.console.print(f"[{self.theme.ERROR}]✗ Failed to connect to {broker_name}[/]")
                return False


class ConfigurationPanel(BaseComponent):
    """Configuration management panel"""
    
    def __init__(self, console: Console = None):
        super().__init__(console)
        self.settings = {}
        
    async def show_settings_menu(self):
        """Show settings configuration menu"""
        self.console.print(f"\n[{self.theme.PRIMARY_GREEN}]=== SYSTEM SETTINGS ===[/]\n", style="bold green")
        
        settings_options = {
            '1': ('Refresh Rate', self.configure_refresh_rate),
            '2': ('Display Theme', self.configure_theme),
            '3': ('Alert Thresholds', self.configure_alerts),
            '4': ('Risk Limits', self.configure_risk_limits),
            '5': ('Data Sources', self.configure_data_sources),
            '6': ('Network Settings', self.configure_network),
            '7': ('Save Configuration', self.save_configuration),
            '8': ('Load Configuration', self.load_configuration),
            '9': ('Reset to Defaults', self.reset_defaults)
        }
        
        for key, (name, _) in settings_options.items():
            self.console.print(f"[{self.theme.PRIMARY_GREEN}]{key}.[/] {name}")
        
        choice = Prompt.ask(
            f"[{self.theme.PRIMARY_GREEN}]Select option[/]",
            choices=list(settings_options.keys())
        )
        
        if choice in settings_options:
            await settings_options[choice][1]()
    
    async def configure_refresh_rate(self):
        """Configure dashboard refresh rate"""
        self.console.print(f"\n[{self.theme.INFO}]Configure Refresh Rate[/]\n")
        
        current_rate = self.settings.get('refresh_rate', 1.0)
        self.console.print(f"Current refresh rate: {current_rate} seconds")
        
        new_rate = FloatPrompt.ask(
            f"[{self.theme.PRIMARY_GREEN}]New refresh rate (0.1-10.0 seconds)[/]",
            default=current_rate,
            choices=[str(i/10) for i in range(1, 101)]
        )
        
        if 0.1 <= new_rate <= 10.0:
            self.settings['refresh_rate'] = new_rate
            self.console.print(f"[{self.theme.SUCCESS}]Refresh rate updated to {new_rate} seconds[/]")
        else:
            self.console.print(f"[{self.theme.ERROR}]Invalid refresh rate[/]")
    
    async def configure_theme(self):
        """Configure display theme"""
        self.console.print(f"\n[{self.theme.INFO}]Configure Theme[/]\n")
        
        themes = ['matrix', 'dark', 'light', 'cyberpunk']
        theme_choice = Prompt.ask(
            f"[{self.theme.PRIMARY_GREEN}]Select theme[/]",
            choices=themes,
            default='matrix'
        )
        
        self.settings['theme'] = theme_choice
        self.console.print(f"[{self.theme.SUCCESS}]Theme updated to {theme_choice}[/]")
    
    async def configure_alerts(self):
        """Configure alert thresholds"""
        self.console.print(f"\n[{self.theme.INFO}]Configure Alert Thresholds[/]\n")
        
        # Daily loss limit
        daily_loss = FloatPrompt.ask(
            f"[{self.theme.PRIMARY_GREEN}]Daily loss limit ($)[/]",
            default=2000.0,
            min=100.0
        )
        
        # Position size limit
        position_size = FloatPrompt.ask(
            f"[{self.theme.PRIMARY_GREEN}]Max position size ($)[/]",
            default=10000.0,
            min=1000.0
        )
        
        # Portfolio exposure limit
        exposure_limit = FloatPrompt.ask(
            f"[{self.theme.PRIMARY_GREEN}]Max portfolio exposure (%[/]",
            default=80.0,
            min=10.0,
            max=100.0
        )
        
        self.settings.update({
            'daily_loss_limit': daily_loss,
            'max_position_size': position_size,
            'max_exposure_pct': exposure_limit
        })
        
        self.console.print(f"[{self.theme.SUCCESS}]Alert thresholds configured[/]")
    
    async def configure_risk_limits(self):
        """Configure risk management limits"""
        self.console.print(f"\n[{self.theme.INFO}]Configure Risk Limits[/]\n")
        
        # Max leverage
        max_leverage = FloatPrompt.ask(
            f"[{self.theme.PRIMARY_GREEN}]Maximum leverage[/]",
            default=3.0,
            min=1.0,
            max=10.0
        )
        
        # Stop loss percentage
        stop_loss_pct = FloatPrompt.ask(
            f"[{self.theme.PRIMARY_GREEN}]Default stop loss (%[/]",
            default=2.0,
            min=0.5,
            max=10.0
        )
        
        # Max concurrent positions
        max_positions = IntPrompt.ask(
            f"[{self.theme.PRIMARY_GREEN}]Max concurrent positions[/]",
            default=10,
            min=1,
            max=50
        )
        
        self.settings.update({
            'max_leverage': max_leverage,
            'default_stop_loss_pct': stop_loss_pct,
            'max_concurrent_positions': max_positions
        })
        
        self.console.print(f"[{self.theme.SUCCESS}]Risk limits configured[/]")
    
    async def configure_data_sources(self):
        """Configure data sources"""
        self.console.print(f"\n[{self.theme.INFO}]Configure Data Sources[/]\n")
        
        data_sources = ['yahoo', 'alpha_vantage', 'polygon', 'iex']
        selected_sources = []
        
        for source in data_sources:
            if Confirm.ask(f"[{self.theme.PRIMARY_GREEN}]Enable {source}?[/]", default=source in ['yahoo']):
                selected_sources.append(source)
        
        self.settings['data_sources'] = selected_sources
        self.console.print(f"[{self.theme.SUCCESS}]Data sources configured: {', '.join(selected_sources)}[/]")
    
    async def configure_network(self):
        """Configure network settings"""
        self.console.print(f"\n[{self.theme.INFO}]Configure Network Settings[/]\n")
        
        # Connection timeout
        timeout = IntPrompt.ask(
            f"[{self.theme.PRIMARY_GREEN}]Connection timeout (seconds)[/]",
            default=30,
            min=5,
            max=300
        )
        
        # Retry attempts
        retries = IntPrompt.ask(
            f"[{self.theme.PRIMARY_GREEN}]Retry attempts[/]",
            default=3,
            min=0,
            max=10
        )
        
        # Request rate limit
        rate_limit = IntPrompt.ask(
            f"[{self.theme.PRIMARY_GREEN}]Request rate limit (requests/minute)[/]",
            default=100,
            min=10,
            max=1000
        )
        
        self.settings.update({
            'connection_timeout': timeout,
            'retry_attempts': retries,
            'rate_limit': rate_limit
        })
        
        self.console.print(f"[{self.theme.SUCCESS}]Network settings configured[/]")
    
    async def save_configuration(self):
        """Save configuration to file"""
        import json
        
        try:
            config_file = "matrix_terminal_config.json"
            with open(config_file, 'w') as f:
                json.dump(self.settings, f, indent=2, default=str)
            
            self.console.print(f"[{self.theme.SUCCESS}]Configuration saved to {config_file}[/]")
        except Exception as e:
            self.console.print(f"[{self.theme.ERROR}]Failed to save configuration: {e}[/]")
    
    async def load_configuration(self):
        """Load configuration from file"""
        import json
        
        try:
            config_file = "matrix_terminal_config.json"
            with open(config_file, 'r') as f:
                self.settings = json.load(f)
            
            self.console.print(f"[{self.theme.SUCCESS}]Configuration loaded from {config_file}[/]")
            self.console.print(f"Loaded {len(self.settings)} settings")
        except FileNotFoundError:
            self.console.print(f"[{self.theme.WARNING}]Configuration file not found[/]")
        except Exception as e:
            self.console.print(f"[{self.theme.ERROR}]Failed to load configuration: {e}[/]")
    
    async def reset_defaults(self):
        """Reset configuration to defaults"""
        if Confirm.ask(f"[{self.theme.WARNING}]Reset all settings to defaults?[/]"):
            self.settings = {
                'refresh_rate': 1.0,
                'theme': 'matrix',
                'daily_loss_limit': 2000.0,
                'max_position_size': 10000.0,
                'max_exposure_pct': 80.0,
                'max_leverage': 3.0,
                'default_stop_loss_pct': 2.0,
                'max_concurrent_positions': 10,
                'data_sources': ['yahoo'],
                'connection_timeout': 30,
                'retry_attempts': 3,
                'rate_limit': 100
            }
            self.console.print(f"[{self.theme.SUCCESS}]Configuration reset to defaults[/]")


class AIChatInterface(BaseComponent):
    """AI chat interface for the trading system"""
    
    def __init__(self, console: Console = None):
        super().__init__(console)
        self.conversation_history = []
        self.chat_active = False
        
    async def start_chat(self):
        """Start AI chat interface"""
        self.console.print(f"\n[{self.theme.PRIMARY_GREEN}]=== AI TRADING ASSISTANT ===[/]\n", style="bold green")
        self.console.print(f"[{self.theme.INFO}]Type your questions about trading, market analysis, or system status[/]")
        self.console.print(f"[{self.theme.WARNING}]Type 'quit' or 'exit' to end chat[/]\n")
        
        self.chat_active = True
        
        # Add welcome message
        welcome_message = {
            'role': 'assistant',
            'content': 'Hello! I am your AI trading assistant. How can I help you today?',
            'timestamp': datetime.now()
        }
        self.conversation_history.append(welcome_message)
        self.console.print(f"[{self.theme.SUCCESS}]Assistant:[/] {welcome_message['content']}\n")
        
        while self.chat_active:
            try:
                user_input = Prompt.ask(f"[{self.theme.PRIMARY_GREEN}]You[/]")
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    break
                
                if not user_input.strip():
                    continue
                
                # Add user message to history
                user_message = {
                    'role': 'user',
                    'content': user_input,
                    'timestamp': datetime.now()
                }
                self.conversation_history.append(user_message)
                
                # Generate AI response (mock for demo)
                response = await self.generate_ai_response(user_input)
                
                # Add AI response to history
                ai_message = {
                    'role': 'assistant',
                    'content': response,
                    'timestamp': datetime.now()
                }
                self.conversation_history.append(ai_message)
                
                # Display response
                self.console.print(f"[{self.theme.SUCCESS}]Assistant:[/] {response}\n")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                self.console.print(f"[{self.theme.ERROR}]Chat error: {e}[/]\n")
        
        self.chat_active = False
        self.console.print(f"[{self.theme.WARNING}]Chat session ended[/]")
    
    async def generate_ai_response(self, user_input: str) -> str:
        """Generate AI response (mock implementation)"""
        # Mock AI responses based on keywords
        input_lower = user_input.lower()
        
        if 'portfolio' in input_lower or 'balance' in input_lower:
            return "Your current portfolio shows a balance of $102,500 with a daily gain of 2.5%. Your largest position is in GOOGL with a $450 unrealized profit."
        
        elif 'market' in input_lower or 'price' in input_lower:
            return "Market is currently showing mixed signals. Technology stocks are leading the gainers, while energy sector is down. Consider monitoring volatility indicators."
        
        elif 'risk' in input_lower or 'drawdown' in input_lower:
            return "Your current risk metrics show a Sharpe ratio of 1.8 and maximum drawdown of -$500. Your portfolio exposure is at 45%, which is within acceptable limits."
        
        elif 'order' in input_lower or 'trade' in input_lower:
            return "You have 3 active orders, including a pending AAPL limit order. Recent trades show a 65% win rate with an average profit of $125 per trade."
        
        elif 'strategy' in input_lower:
            return "Your Trend Follower strategy is currently active and generating profits. Consider reviewing parameters for Mean Reversion strategy which is currently paused."
        
        elif 'help' in input_lower or 'what can' in input_lower:
            return "I can help you with portfolio analysis, market insights, risk assessment, order status, strategy recommendations, and general trading questions. What would you like to know?"
        
        elif 'hello' in input_lower or 'hi' in input_lower:
            return "Hello! I'm here to help with your trading decisions. You can ask me about your portfolio, market conditions, risk metrics, or any trading-related questions."
        
        else:
            # Generate contextual response
            responses = [
                "That's an interesting question. Based on current market conditions and your portfolio, I'd recommend monitoring volatility indicators.",
                "Let me analyze that for you. Your current positions show good diversification across technology stocks.",
                "I can help you with that. From a risk management perspective, consider your current exposure levels.",
                "That's a good point. Market sentiment is currently neutral with mixed sector performance."
            ]
            
            import random
            return random.choice(responses)
    
    def get_chat_history(self) -> List[Dict]:
        """Get conversation history"""
        return self.conversation_history.copy()
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history.clear()
        self.console.print(f"[{self.theme.INFO}]Chat history cleared[/]")