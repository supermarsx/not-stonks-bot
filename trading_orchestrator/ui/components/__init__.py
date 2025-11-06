"""
Matrix UI Components Package
Complete set of Matrix-themed terminal interface components
"""

# Import base components
from .base_components import (
    BaseComponent,
    PanelComponent, 
    TableComponent,
    ProgressComponent,
    InteractiveComponent,
    AlertComponent
)

# Import trading-specific components  
from .trading_components import (
    PortfolioComponent,
    OrderComponent,
    MarketDataComponent,
    RiskComponent,
    StrategyComponent,
    BrokerComponent
)

# Import real-time components
from .realtime_components import (
    StreamProcessor,
    LivePortfolioTracker,
    LiveMarketTicker,
    LiveOrderMonitor,
    LiveRiskMonitor,
    RealTimeDataManager,
    LoadingIndicator
)

# Import theme
from ..themes.matrix_theme import MatrixTheme, MatrixEffects

# Package metadata
__version__ = "1.0.0"
__author__ = "Day Trading Orchestrator Team"

# Convenience exports
__all__ = [
    # Base components
    "BaseComponent",
    "PanelComponent", 
    "TableComponent",
    "ProgressComponent",
    "InteractiveComponent",
    "AlertComponent",
    
    # Trading components
    "PortfolioComponent",
    "OrderComponent", 
    "MarketDataComponent",
    "RiskComponent",
    "StrategyComponent",
    "BrokerComponent",
    
    # Real-time components
    "StreamProcessor",
    "LivePortfolioTracker",
    "LiveMarketTicker", 
    "LiveOrderMonitor",
    "LiveRiskMonitor",
    "RealTimeDataManager",
    "LoadingIndicator",
    
    # Theme
    "MatrixTheme",
    "MatrixEffects"
]
