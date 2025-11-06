# Matrix-Themed Terminal Interface

A complete, production-ready Matrix-themed terminal interface for the Day Trading Orchestrator system. Built with Rich, Textual, and custom Matrix theming for an immersive trading experience.

## 🎯 Overview

The Matrix Terminal Interface provides a complete, real-time trading dashboard with:

- **Matrix-themed UI** - Green-on-black terminal interface with ASCII art
- **Real-time updates** - Live portfolio tracking, market data, and order monitoring  
- **Interactive components** - Order entry, broker setup, configuration panels
- **Risk management** - Real-time risk monitoring and alert system
- **Strategy management** - Trading strategy selection and configuration
- **AI chat integration** - Interactive AI assistant for trading insights
- **Multi-broker support** - Connection management for multiple broker APIs

## 🏗️ Architecture

```
trading_orchestrator/ui/
├── interface.py              # Main terminal interface
├── demo.py                   # Comprehensive demo script
├── themes/
│   └── matrix_theme.py       # Matrix color palette and styling
└── components/
    ├── __init__.py           # Package exports
    ├── base_components.py    # Reusable UI components
    ├── trading_components.py # Trading-specific components
    ├── realtime_components.py # Real-time data streaming
    └── interactive_elements.py # Forms and setup wizards
```

## 🎨 Matrix Theme

The interface uses a custom Matrix color palette:

- **Primary Green** (`bright_green`) - Main UI elements and headers
- **Matrix Cyan** (`bright_cyan`) - Accent elements and code
- **Success Green** - Positive values, confirmations
- **Warning Yellow** - Alerts and cautions
- **Error Red** - Errors and losses
- **White** - Primary text content

### Key Features:
- ASCII art headers and dividers
- Matrix-style borders and panels
- Typewriter effects and blinking text
- Color-coded data visualization
- Matrix rain effects

## 📦 Components

### Base Components (`base_components.py`)

- **BaseComponent** - Base class for all UI components
- **PanelComponent** - Matrix-styled panel creation
- **TableComponent** - Data tables with automatic formatting
- **ProgressComponent** - Progress bars and loading indicators
- **InteractiveComponent** - User input and navigation
- **AlertComponent** - Notifications and alerts

### Trading Components (`trading_components.py`)

- **PortfolioComponent** - Account balance and equity tracking
- **OrderComponent** - Order book and trade history display
- **MarketDataComponent** - Market ticker and price charts
- **RiskComponent** - Risk metrics and alerts
- **StrategyComponent** - Strategy selection and configuration
- **BrokerComponent** - Connection status and management

### Real-time Components (`realtime_components.py`)

- **LivePortfolioTracker** - Real-time portfolio updates
- **LiveMarketTicker** - Streaming market data
- **LiveOrderMonitor** - Order execution tracking
- **LiveRiskMonitor** - Real-time risk metrics
- **RealTimeDataManager** - Unified data stream management
- **LoadingIndicator** - Matrix-styled loading states

### Interactive Elements (`interactive_elements.py`)

- **FormComponent** - Dynamic form creation and validation
- **OrderEntryForm** - Interactive order entry wizard
- **BrokerSetupWizard** - Broker configuration setup
- **ConfigurationPanel** - System settings management
- **AIChatInterface** - Interactive AI assistant

## 🚀 Usage

### Basic Dashboard

```python
from ui.interface import MatrixTerminalInterface

# Create and run dashboard
interface = MatrixTerminalInterface()
await interface.run("dashboard")  # Real-time dashboard
```

### Interactive Mode

```python
# Run interactive command mode
await interface.run("interactive")  # Menu-driven interface
```

### Individual Components

```python
from ui.components import PortfolioComponent, MarketDataComponent

# Create portfolio component
portfolio_comp = PortfolioComponent(console)
portfolio_panel = portfolio_comp.create_portfolio_summary({
    'balance': 100000,
    'equity': 105000,
    'day_change': 5000
})

# Create market ticker
market_comp = MarketDataComponent(console)
ticker_panel = market_comp.create_ticker(['AAPL', 'TSLA', 'GOOGL'])
```

### Real-time Data

```python
from ui.components.realtime_components import RealTimeDataManager

# Start real-time data streams
realtime_manager = RealTimeDataManager(console)
await realtime_manager.start_all_streams()

# Get dashboard components
components = realtime_manager.get_dashboard_components()
```

## 🎛️ Interactive Features

### Order Entry Form

```python
from ui.components.interactive_elements import OrderEntryForm

order_form = OrderEntryForm(console)
order = await order_form.get_order_input()
# Returns: {
#     'order_id': 'ORD123',
#     'symbol': 'AAPL',
#     'side': 'BUY',
#     'quantity': 100,
#     'order_type': 'LIMIT',
#     'price': 150.00
# }
```

### Broker Setup Wizard

```python
from ui.components.interactive_elements import BrokerSetupWizard

broker_setup = BrokerSetupWizard(console)
config = await broker_setup.setup_broker('alpaca')
success = await broker_setup.test_connection(config)
```

### Configuration Panel

```python
from ui.components.interactive_elements import ConfigurationPanel

config_panel = ConfigurationPanel(console)
await config_panel.show_settings_menu()
# Interactive settings configuration
```

### AI Chat Interface

```python
from ui.components.interactive_elements import AIChatInterface

ai_chat = AIChatInterface(console)
await ai_chat.start_chat()
# Interactive AI assistant for trading insights
```

## 🎮 Demo

Run the comprehensive demo to see all components in action:

```bash
cd trading_orchestrator
python -m ui.demo
```

The demo includes:
- Theme showcase with color palette
- Portfolio component demonstration
- Trading interface walkthrough
- Risk management display
- Strategy selection interface
- Broker management panel
- Real-time data streaming
- Interactive form examples
- Full dashboard demonstration

## ⚡ Real-time Features

### Live Dashboard Updates
- Portfolio value updates every second
- Market ticker with price changes
- Order status monitoring
- Risk metrics tracking
- Connection status display

### Streaming Data
- WebSocket integration for real-time feeds
- Buffer management for data streams
- Subscriber pattern for component updates
- Automatic UI refresh without flicker

### Error Handling
- Graceful degradation on connection loss
- Automatic reconnection attempts
- User-friendly error messages
- System health monitoring

## 🔧 Integration

### With Trading Orchestrator

```python
from ui.interface import get_matrix_terminal
from brokers.factory import BrokerFactory

# Get terminal instance
terminal = get_matrix_terminal()

# Connect to broker data
async def portfolio_data_source():
    broker = BrokerFactory.get_broker('alpaca')
    account = await broker.get_account()
    return {
        'balance': account.balance,
        'equity': account.equity,
        'buying_power': account.buying_power
    }

# Start dashboard with real data
await terminal.realtime_manager.start_all_streams(portfolio_data_source)
```

### With AI System

```python
from ai.orchestrator import TradingAI

# Connect AI responses to chat interface
ai = TradingAI()

# In AI chat component
async def generate_response(user_input):
    response = await ai.analyze_query(user_input)
    return response.content
```

## 🎯 Keyboard Shortcuts

- **Q** - Quit application
- **R** - Refresh dashboard
- **D** - Dashboard view
- **O** - Orders management
- **P** - Portfolio details
- **S** - Settings panel
- **?** - Help menu
- **ESC** - Return to dashboard

## 📊 Data Format

### Portfolio Data
```python
{
    'balance': float,
    'equity': float,
    'buying_power': float,
    'day_change': float,
    'day_change_pct': float
}
```

### Position Data
```python
{
    'symbol': str,
    'side': str,  # 'LONG' or 'SHORT'
    'quantity': float,
    'current_price': float,
    'unrealized_pnl': float,
    'realized_pnl': float
}
```

### Order Data
```python
{
    'order_id': str,
    'symbol': str,
    'order_type': str,
    'side': str,
    'quantity': float,
    'price': float,
    'status': str,
    'timestamp': datetime
}
```

### Risk Data
```python
{
    'portfolio_value': float,
    'daily_pnl': float,
    'max_drawdown': float,
    'sharpe_ratio': float,
    'win_rate': float,
    'open_positions': int,
    'leverage': float,
    'risk_score': str
}
```

## 🎨 Customization

### Theme Colors

```python
from ui.themes.matrix_theme import MatrixTheme

# Customize color palette
theme = MatrixTheme()
theme.PRIMARY_GREEN = "chartreuse"
theme.ERROR = "orange_red1"

# Create custom styles
custom_styles = theme.get_matrix_styles()
custom_styles['custom_element'] = Style(color="purple", bold=True)
```

### Component Styling

```python
from rich.style import Style

# Custom panel border
custom_border = Style(color="bright_blue", bold=True)

# Custom table styling
custom_table = theme.create_matrix_table()
custom_table.header_style = "bright_magenta"
```

## 🔒 Security Features

- Password masking for sensitive inputs
- Secure credential storage for broker API keys
- Input validation and sanitization
- Secure connection testing
- Error message sanitization

## 📈 Performance

- Efficient rendering with Rich library
- Optimized real-time updates
- Memory-efficient data buffering
- Smooth animations and transitions
- Responsive design for various terminal sizes

## 🐛 Error Handling

- Graceful degradation on component failures
- User-friendly error messages
- Automatic recovery mechanisms
- Comprehensive logging
- Health check endpoints

## 🚀 Future Enhancements

- Textual-based full-screen applications
- Custom widget development
- Plugin architecture for components
- Advanced charting capabilities
- Voice commands integration
- Mobile terminal support
- Custom theme editor
- Component marketplace

## 📋 Requirements

```
rich>=13.7.0
textual>=0.44.1
blessed>=1.20.0
prompt-toolkit>=3.0.43
```

## 🤝 Contributing

1. Follow Matrix theme guidelines
2. Maintain component modularity
3. Add comprehensive documentation
4. Include unit tests for new components
5. Ensure backward compatibility

## 📄 License

Part of the Day Trading Orchestrator system. See main project license for details.

---

**Matrix Terminal Interface v1.0**  
*The future of terminal-based trading interfaces*
