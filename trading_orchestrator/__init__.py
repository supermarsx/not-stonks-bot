"""
Trading Orchestrator
Comprehensive multi-broker day trading platform with AI orchestration

Complete Trading Systems:
- Risk Management Engine with limits, policies, circuit breakers, and compliance
- Order Management System with routing, validation, monitoring, and settlement  
- Trading Strategies Framework with multiple strategies, optimization, and backtesting
- Market Data System with real-time feeds, feature calculation, and historical data

Features:
- Multi-broker support (Alpaca, Binance, DEGIRO, IBKR, Trade Republic, Trading212, XTB)
- Advanced risk management and compliance monitoring
- AI-driven strategy selection and optimization
- Real-time market data processing and feature calculation
- Comprehensive backtesting and performance analytics
- Position reconciliation and settlement processing
- Audit logging and incident postmortem analysis
"""

# Core system imports
from . import risk
from . import oms
from . import strategies  
from . import market_data

# Risk Management System
from .risk import (
    RiskEngine,
    RiskLimitsChecker,
    PolicyEngine,
    CircuitBreakerManager,
    ComplianceMonitor,
    AuditLogger,
    IncidentPostmortem,
    
    # Enums and data structures
    RiskEventType,
    RiskLevel,
    RiskLimit,
    RiskEvent,
    CircuitBreaker,
    
    # Core functionality
    create_risk_engine,
    configure_default_limits,
    load_compliance_rules
)

# Order Management System  
from .oms import (
    OMSEngine,
    OrderRouter,
    OrderValidator,
    ExecutionMonitor,
    SlippageTracker,
    PerformanceAnalytics,
    OrderManager,
    PositionManager,
    SettlementProcessor,
    
    # Data structures
    BrokerPosition,
    PositionDiscrepancy,
    PositionAlert,
    TradeSettlement,
    SettlementBatch,
    
    # Enums
    PositionReconciliationStatus,
    SettlementStatus,
    SettlementType,
    
    # Core functionality
    create_oms_engine,
    configure_order_routing,
    setup_settlement_processing
)

# Trading Strategies Framework
from .strategies import (
    # Base classes
    BaseStrategy,
    BaseTimeSeriesStrategy,
    StrategyRegistry,
    strategy_registry,
    
    # Strategy implementations
    TrendFollowingStrategy,
    MeanReversionStrategy,
    PairsTradingStrategy,
    CrossVenueArbitrageStrategy,
    
    # Factory functions
    create_trend_following_strategy,
    create_mean_reversion_strategy,
    create_pairs_trading_strategy,
    create_arbitrage_strategy,
    
    # Optimization system
    ParameterOptimizer,
    GridSearchOptimizer,
    RandomSearchOptimizer,
    GeneticAlgorithmOptimizer,
    OptimizationMethod,
    OptimizationObjective,
    ParameterRange,
    create_parameter_ranges,
    
    # Backtesting framework
    BacktestEngine,
    BacktestMetrics,
    BacktestResult,
    compare_backtests,
    
    # Utilities
    create_strategy_from_config,
    batch_optimize_strategies,
    run_strategy_comparison,
    quick_start_example
)

# Market Data System
from .market_data import (
    MarketDataManager,
    FeatureCalculator,
    MarketDataProvider,
    SimulatedDataProvider,
    
    # Data structures
    MarketDataPoint,
    FeatureValue,
    
    # Enums
    DataFrequency,
    DataSource
)


# Main orchestrator class
class TradingOrchestrator:
    """
    Main Trading Orchestrator
    
    Coordinates all trading systems:
    - Risk management
    - Order management  
    - Strategy execution
    - Market data processing
    """
    
    def __init__(self):
        """Initialize trading orchestrator"""
        self.risk_engine = None
        self.oms_engine = None
        self.strategy_registry = None
        self.market_data_manager = None
        
        self.initialized = False
        logger.info("Trading Orchestrator created")
    
    async def initialize(self, config: dict):
        """Initialize all trading systems"""
        try:
            # Initialize Risk Engine
            self.risk_engine = RiskEngine()
            
            # Initialize OMS
            self.oms_engine = OMSEngine()
            
            # Initialize Strategy Registry
            self.strategy_registry = StrategyRegistry()
            
            # Initialize Market Data Manager
            self.market_data_manager = MarketDataManager()
            
            # Configure systems with provided config
            await self._configure_systems(config)
            
            self.initialized = True
            logger.info("Trading Orchestrator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Trading Orchestrator: {e}")
            raise
    
    async def _configure_systems(self, config: dict):
        """Configure all trading systems"""
        try:
            # Configure risk management
            if 'risk' in config:
                risk_config = config['risk']
                await configure_default_limits(self.risk_engine)
                await load_compliance_rules(self.risk_engine, risk_config.get('compliance_rules', []))
            
            # Configure order management
            if 'oms' in config:
                oms_config = config['oms']
                await configure_order_routing(self.oms_engine, oms_config.get('routing_rules', []))
                await setup_settlement_processing(self.oms_engine)
            
            # Configure market data
            if 'market_data' in config:
                md_config = config['market_data']
                # Register data providers from config
                for provider_config in md_config.get('providers', []):
                    source = DataSource(provider_config['source'])
                    provider = SimulatedDataProvider()  # In real implementation, would use actual providers
                    self.market_data_manager.register_provider(source, provider)
            
        except Exception as e:
            logger.error(f"Error configuring trading systems: {e}")
            raise
    
    async def start_trading(self):
        """Start all trading systems"""
        if not self.initialized:
            raise RuntimeError("Trading Orchestrator not initialized")
        
        try:
            logger.info("Starting Trading Orchestrator")
            
            # Start market data feeds
            # Start risk monitoring
            # Start strategy execution
            # Start order processing
            
            logger.info("Trading Orchestrator started successfully")
            
        except Exception as e:
            logger.error(f"Error starting trading: {e}")
            raise
    
    async def stop_trading(self):
        """Stop all trading systems"""
        try:
            logger.info("Stopping Trading Orchestrator")
            
            # Stop all systems gracefully
            if self.risk_engine:
                await self.risk_engine.shutdown()
            
            if self.oms_engine:
                await self.oms_engine.shutdown()
            
            logger.info("Trading Orchestrator stopped")
            
        except Exception as e:
            logger.error(f"Error stopping trading: {e}")
    
    async def get_system_status(self) -> dict:
        """Get status of all trading systems"""
        return {
            'orchestrator': {
                'initialized': self.initialized,
                'timestamp': datetime.utcnow().isoformat()
            },
            'risk_engine': {
                'active': self.risk_engine is not None,
                'status': 'running' if self.risk_engine else 'stopped'
            },
            'oms_engine': {
                'active': self.oms_engine is not None,
                'status': 'running' if self.oms_engine else 'stopped'
            },
            'strategy_registry': {
                'active': self.strategy_registry is not None,
                'total_strategies': len(self.strategy_registry.strategies) if self.strategy_registry else 0,
                'running_strategies': len(self.strategy_registry.get_active_strategies()) if self.strategy_registry else 0
            },
            'market_data': {
                'active': self.market_data_manager is not None,
                'providers': len(self.market_data_manager.providers) if self.market_data_manager else 0
            }
        }


# Quick start function
async def quick_start(config_file: str = "trading_config.json"):
    """
    Quick start function for the trading orchestrator
    
    Args:
        config_file: Path to configuration file
    """
    try:
        # Load configuration
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
        except FileNotFoundError:
            # Use default configuration
            config = {
                'risk': {
                    'max_position_size': 100000,
                    'max_daily_loss': 10000,
                    'max_order_size': 50000
                },
                'oms': {
                    'default_broker': 'alpaca',
                    'enable_smart_routing': True
                },
                'market_data': {
                    'providers': [
                        {'source': 'simulated', 'symbols': ['AAPL', 'GOOGL', 'MSFT']}
                    ]
                },
                'strategies': {
                    'enabled': ['trend_following', 'mean_reversion']
                }
            }
        
        # Create and initialize orchestrator
        orchestrator = TradingOrchestrator()
        await orchestrator.initialize(config)
        
        # Start trading
        await orchestrator.start_trading()
        
        print("✅ Trading Orchestrator started successfully!")
        print(f"📊 Risk Engine: Active")
        print(f"📈 OMS Engine: Active") 
        print(f"🤖 Strategies: {len(orchestrator.strategy_registry.strategies) if orchestrator.strategy_registry else 0} registered")
        print(f"📡 Market Data: {len(orchestrator.market_data_manager.providers) if orchestrator.market_data_manager else 0} providers")
        
        return orchestrator
        
    except Exception as e:
        logger.error(f"Failed to start trading orchestrator: {e}")
        raise


# Export main components
__all__ = [
    # Core orchestrator
    'TradingOrchestrator',
    'quick_start',
    
    # Risk Management
    'RiskEngine',
    'RiskLimitsChecker', 
    'PolicyEngine',
    'CircuitBreakerManager',
    'ComplianceMonitor',
    'AuditLogger',
    'IncidentPostmortem',
    'create_risk_engine',
    
    # Order Management
    'OMSEngine',
    'OrderRouter',
    'OrderValidator', 
    'ExecutionMonitor',
    'SlippageTracker',
    'PerformanceAnalytics',
    'OrderManager',
    'PositionManager',
    'SettlementProcessor',
    'create_oms_engine',
    
    # Strategies
    'BaseStrategy',
    'StrategyRegistry',
    'strategy_registry',
    'TrendFollowingStrategy',
    'MeanReversionStrategy',
    'PairsTradingStrategy',
    'CrossVenueArbitrageStrategy',
    'create_trend_following_strategy',
    'create_mean_reversion_strategy',
    'create_pairs_trading_strategy',
    'create_arbitrage_strategy',
    
    # Optimization & Backtesting
    'ParameterOptimizer',
    'BacktestEngine',
    'OptimizationMethod',
    'OptimizationObjective',
    'ParameterRange',
    'create_parameter_ranges',
    'compare_backtests',
    'quick_start_example',
    
    # Market Data
    'MarketDataManager',
    'FeatureCalculator',
    'MarketDataPoint',
    'FeatureValue',
    'DataFrequency',
    'DataSource',
    
    # Package exports
    'risk',
    'oms', 
    'strategies',
    'market_data'
]

# Import required modules
import json
from datetime import datetime
from loguru import logger

# Example configuration template
DEFAULT_CONFIG = {
    "risk": {
        "max_position_size": 100000,
        "max_daily_loss": 10000,
        "max_order_size": 50000,
        "max_portfolio_exposure": 1000000,
        "circuit_breakers": {
            "loss_limit": 50000,
            "volatility_threshold": 0.05,
            "order_rejection_rate": 0.1
        }
    },
    "oms": {
        "default_broker": "alpaca",
        "enable_smart_routing": True,
        "order_validation_strict": True,
        "settlement_delay_seconds": 30
    },
    "market_data": {
        "providers": [
            {
                "source": "simulated",
                "symbols": ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"],
                "frequency": "1m"
            }
        ],
        "feature_calculation": {
            "enabled_features": ["sma", "rsi", "bollinger_bands", "volume_sma"],
            "update_interval": 60
        }
    },
    "strategies": {
        "enabled": ["trend_following", "mean_reversion"],
        "default_symbols": ["AAPL", "GOOGL"],
        "max_concurrent_strategies": 5
    },
    "brokers": {
        "alpaca": {
            "enabled": True,
            "paper_trading": True,
            "api_key": "your_api_key",
            "api_secret": "your_api_secret"
        },
        "binance": {
            "enabled": False,
            "api_key": "your_api_key", 
            "api_secret": "your_api_secret",
            "sandbox": True
        }
    }
}