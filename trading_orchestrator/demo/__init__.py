"""
Demo Mode Package for Trading Simulation

Complete demo mode implementation for safe simulation trading without real money.
Allows testing strategies, validating risk management, and training without financial risk.
"""

# Core demo mode management
from .demo_mode_manager import (
    DemoModeManager,
    DemoModeConfig,
    DemoEnvironment,
    DemoModeState,
    demo_manager
)

# Virtual trading infrastructure
from .virtual_broker import (
    VirtualBroker,
    VirtualOrder,
    VirtualPosition,
    OrderStatus,
    FillType,
    ExecutionResult
)

from .paper_trading_engine import (
    PaperTradingEngine,
    OrderBookLevel,
    MarketImpact,
    ExecutionPlan,
    ExecutionSlice
)

from .virtual_portfolio import (
    VirtualPortfolio,
    PortfolioPosition,
    PortfolioTrade,
    PortfolioSnapshot,
    RiskMetrics
)

# Analytics and monitoring
from .demo_logging import (
    DemoLogger,
    TradeLogEntry,
    PerformanceLogEntry,
    LogLevel
)

from .demo_backtesting import (
    DemoBacktester,
    BacktestResults,
    BacktestSnapshot,
    StrategyPerformance
)

from .demo_dashboard import (
    DemoDashboard,
    DashboardWidget,
    Alert,
    DashboardData
)

# Validation and risk management
from .demo_validator import (
    RiskSimulator,
    ValidationResult,
    RiskParameter,
    StressTestResult
)

from .demo_validator_tool import (
    DemoModeValidator,
    ValidationTestResult,
    ValidationSuiteResult
)

# Integration and UI components
from .demo_broker_factory import (
    DemoBrokerFactory,
    DemoDataGenerator,
    DemoDataConfig
)

from .demo_ui_components import (
    DemoModeUI
)

# All public exports
__all__ = [
    # Core management
    'DemoModeManager',
    'DemoModeConfig',
    'DemoEnvironment', 
    'DemoModeState',
    'demo_manager',
    
    # Virtual trading
    'VirtualBroker',
    'VirtualOrder',
    'VirtualPosition',
    'OrderStatus',
    'FillType',
    'ExecutionResult',
    'PaperTradingEngine',
    'OrderBookLevel',
    'MarketImpact',
    'ExecutionPlan',
    'ExecutionSlice',
    'VirtualPortfolio',
    'PortfolioPosition',
    'PortfolioTrade',
    'PortfolioSnapshot',
    'RiskMetrics',
    
    # Analytics
    'DemoLogger',
    'TradeLogEntry',
    'PerformanceLogEntry',
    'LogLevel',
    'DemoBacktester',
    'BacktestResults',
    'BacktestSnapshot',
    'StrategyPerformance',
    'DemoDashboard',
    'DashboardWidget',
    'Alert',
    'DashboardData',
    
    # Validation
    'RiskSimulator',
    'ValidationResult',
    'RiskParameter',
    'StressTestResult',
    'DemoModeValidator',
    'ValidationTestResult',
    'ValidationSuiteResult',
    
    # Integration
    'DemoBrokerFactory',
    'DemoDataGenerator',
    'DemoDataConfig',
    'DemoModeUI'
]
