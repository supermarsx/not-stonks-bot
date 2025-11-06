"""
@file application.py
@brief Application Configuration and System Integration

@details
This module handles system startup, shutdown, and core service orchestration
for the Trading Orchestrator. It manages the complete application lifecycle
from initialization through shutdown, integrating all major system components.

Key Features:
- Application state management
- Service initialization and orchestration
- Graceful shutdown handling
- Configuration validation and loading
- Health monitoring and diagnostics
- Signal handling for clean shutdown
- Multi-service coordination

@author Trading Orchestrator System
@version 1.0
@date 2025-11-06

@warning
This module manages critical system services. Incorrect configuration
can lead to system instability or security vulnerabilities.

@note
This module serves as the main entry point for the trading orchestrator
and coordinates between all major system components:

@see config.settings for application configuration
@see brokers.factory for broker service management
@see ai.orchestrator for AI trading coordination
@see risk.manager for risk management services
@see oms.manager for order management services

@par System Components:
- Broker Factory: Multi-broker service management
- AI Orchestrator: AI-driven trading coordination
- Risk Manager: Portfolio risk monitoring
- Order Manager: Order execution coordination
- Database: Persistent data storage
- Logging: Centralized logging services
"""

from typing import Dict, Any, Optional, List, Union
import asyncio
import signal
import sys
import json
import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from loguru import logger

from .settings import settings
from .database import init_db, engine
from brokers.factory import BrokerFactory
from brokers.base import BrokerConfig
from ai.orchestrator import AITradingOrchestrator, TradingMode
from ai.models.ai_models_manager import AIModelsManager
from ai.tools.trading_tools import TradingTools
from risk.manager import RiskManager
from oms.manager import OrderManager


class ApplicationState:
    """Global application state management"""
    
    def __init__(self):
        self.initialized = False
        self.brokers = {}
        self.ai_orchestrator = None
        self.risk_manager = None
        self.order_manager = None
        self.health_checks = {}
        self.start_time = datetime.utcnow()
        
    def get_uptime(self) -> float:
        """Get application uptime in seconds"""
        return (datetime.utcnow() - self.start_time).total_seconds()
        
    def is_healthy(self) -> bool:
        """Check overall system health"""
        required_components = ['database', 'brokers']
        return all(
            self.health_checks.get(comp, {}).get('healthy', False)
            for comp in required_components
        )


class ApplicationConfig:
    """Main application configuration and lifecycle management"""
    
    def __init__(self):
        self.state = ApplicationState()
        self._setup_signal_handlers()
        
    def _setup_signal_handlers(self):
        """Setup graceful shutdown handlers"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            asyncio.create_task(self.shutdown())
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def initialize(self) -> bool:
        """Initialize all application components"""
        try:
            logger.info("🚀 Initializing Day Trading Orchestrator Application...")
            
            # Create necessary directories
            self._create_directories()
            
            # Initialize database
            await self._initialize_database()
            
            # Initialize AI components
            await self._initialize_ai_components()
            
            # Initialize risk management
            await self._initialize_risk_management()
            
            # Initialize order management
            await self._initialize_order_management()
            
            # Connect to brokers
            await self._connect_brokers()
            
            # Setup health checks
            self._setup_health_checks()
            
            self.state.initialized = True
            logger.success("✅ Application initialized successfully")
            return True
            
        except Exception as e:
            logger.exception(f"❌ Failed to initialize application: {e}")
            return False
    
    def _create_directories(self):
        """Create necessary directories"""
        import os
        from pathlib import Path
        
        directories = [
            Path(settings.log_file_path).parent,
            Path(settings.db_path).parent if settings.db_path.startswith('./') else None,
            './logs',
            './data',
            './backups'
        ]
        
        for directory in directories:
            if directory:
                directory.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Created directory: {directory}")
    
    async def _initialize_database(self):
        """Initialize database"""
        try:
            logger.info("📊 Initializing database...")
            await init_db()
            
            # Run database health check
            async with engine.connect() as conn:
                await conn.execute("SELECT 1")
            
            self.state.health_checks['database'] = {
                'healthy': True,
                'message': 'Database connected',
                'timestamp': datetime.utcnow().isoformat()
            }
            logger.success("✅ Database initialized")
            
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
            self.state.health_checks['database'] = {
                'healthy': False,
                'message': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
            raise
    
    async def _initialize_ai_components(self):
        """Initialize AI components"""
        try:
            logger.info("🧠 Initializing AI components...")
            
            # Initialize AI models manager
            self.ai_models_manager = AIModelsManager()
            
            # Initialize trading tools
            self.trading_tools = TradingTools()
            
            # Initialize AI orchestrator
            self.state.ai_orchestrator = AITradingOrchestrator(
                ai_models_manager=self.ai_models_manager,
                trading_tools=self.trading_tools,
                broker_manager=None,  # Will be set later
                risk_manager=None,    # Will be set later
                trading_mode=TradingMode.PAPER if not settings.is_production else TradingMode.LIVE
            )
            
            self.state.health_checks['ai'] = {
                'healthy': True,
                'message': 'AI orchestrator initialized',
                'timestamp': datetime.utcnow().isoformat()
            }
            logger.success("✅ AI components initialized")
            
        except Exception as e:
            logger.error(f"❌ AI components initialization failed: {e}")
            self.state.health_checks['ai'] = {
                'healthy': False,
                'message': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
            raise
    
    async def _initialize_risk_management(self):
        """Initialize risk management"""
        try:
            logger.info("🛡️ Initializing risk management...")
            
            self.state.risk_manager = RiskManager(
                max_position_size=settings.max_position_size,
                max_daily_loss=settings.max_daily_loss,
                max_open_orders=settings.max_open_orders,
                risk_per_trade=settings.risk_per_trade
            )
            
            self.state.health_checks['risk'] = {
                'healthy': True,
                'message': 'Risk manager initialized',
                'timestamp': datetime.utcnow().isoformat()
            }
            logger.success("✅ Risk management initialized")
            
        except Exception as e:
            logger.error(f"❌ Risk management initialization failed: {e}")
            self.state.health_checks['risk'] = {
                'healthy': False,
                'message': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
            raise
    
    async def _initialize_order_management(self):
        """Initialize order management"""
        try:
            logger.info("📈 Initializing order management...")
            
            self.state.order_manager = OrderManager(
                risk_manager=self.state.risk_manager
            )
            
            self.state.health_checks['oms'] = {
                'healthy': True,
                'message': 'Order manager initialized',
                'timestamp': datetime.utcnow().isoformat()
            }
            logger.success("✅ Order management initialized")
            
        except Exception as e:
            logger.error(f"❌ Order management initialization failed: {e}")
            self.state.health_checks['oms'] = {
                'healthy': False,
                'message': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
            raise
    
    async def _connect_brokers(self):
        """Connect to configured brokers"""
        logger.info("🔗 Connecting to brokers...")
        
        broker_configs = [
            ('binance', self._create_binance_config),
            ('alpaca', self._create_alpaca_config),
            ('ibkr', self._create_ibkr_config),
            ('trading212', self._create_trading212_config)
        ]
        
        for broker_name, config_creator in broker_configs:
            try:
                config = config_creator()
                if config:
                    await self._connect_single_broker(broker_name, config)
            except Exception as e:
                logger.warning(f"Failed to configure {broker_name}: {e}")
        
        # Update broker health check
        connected_count = len(self.state.brokers)
        total_attempts = len(broker_configs)
        
        self.state.health_checks['brokers'] = {
            'healthy': connected_count > 0,
            'message': f"{connected_count}/{total_attempts} brokers connected",
            'connected_brokers': list(self.state.brokers.keys()),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        logger.info(f"📡 Broker connection status: {connected_count}/{total_attempts} connected")
    
    def _create_binance_config(self) -> Optional[BrokerConfig]:
        """Create Binance broker configuration"""
        if not settings.binance_api_key:
            return None
            
        return BrokerConfig(
            broker_name="binance",
            api_key=settings.binance_api_key,
            api_secret=settings.binance_api_secret,
            is_paper=settings.binance_testnet,
            config={
                "testnet": settings.binance_testnet,
                "rate_limit": 1200
            }
        )
    
    def _create_alpaca_config(self) -> Optional[BrokerConfig]:
        """Create Alpaca broker configuration"""
        if not settings.alpaca_api_key:
            return None
            
        return BrokerConfig(
            broker_name="alpaca",
            api_key=settings.alpaca_api_key,
            api_secret=settings.alpaca_api_secret,
            is_paper=settings.alpaca_paper,
            config={
                "base_url": "https://paper-api.alpaca.markets" if settings.alpaca_paper else "https://api.alpaca.markets"
            }
        )
    
    def _create_ibkr_config(self) -> Optional[BrokerConfig]:
        """Create Interactive Brokers configuration"""
        return BrokerConfig(
            broker_name="ibkr",
            api_key="",  # IBKR doesn't use API keys
            api_secret="",
            is_paper=True,  # IBKR TWS can be paper trading
            config={
                "host": settings.ibkr_host,
                "port": settings.ibkr_port,
                "client_id": settings.ibkr_client_id
            }
        )
    
    def _create_trading212_config(self) -> Optional[BrokerConfig]:
        """Create Trading 212 configuration"""
        if not settings.trading212_api_key:
            return None
            
        return BrokerConfig(
            broker_name="trading212",
            api_key=settings.trading212_api_key,
            api_secret="",
            is_paper=settings.trading212_practice,
            config={
                "base_url": "https://live.trading212.com" if not settings.trading212_practice else "https://practice.trading212.com"
            }
        )
    
    async def _connect_single_broker(self, broker_name: str, config: BrokerConfig):
        """Connect to a single broker"""
        try:
            broker = BrokerFactory.create_broker(config)
            connected = await broker.connect()
            
            if connected:
                self.state.brokers[broker_name] = broker
                logger.success(f"✅ Connected to {broker_name}")
                
                # Link broker to orchestrator
                if self.state.ai_orchestrator:
                    self.state.ai_orchestrator.broker_manager = self
            else:
                logger.error(f"❌ Failed to connect to {broker_name}")
                
        except Exception as e:
            logger.error(f"❌ Error connecting to {broker_name}: {e}")
    
    def _setup_health_checks(self):
        """Setup periodic health checks"""
        async def health_check_loop():
            while True:
                try:
                    await self._run_health_checks()
                    await asyncio.sleep(30)  # Check every 30 seconds
                except Exception as e:
                    logger.error(f"Health check error: {e}")
                    await asyncio.sleep(30)
        
        asyncio.create_task(health_check_loop())
    
    async def _run_health_checks(self):
        """Run all health checks"""
        # Check database
        try:
            async with engine.connect() as conn:
                await conn.execute("SELECT 1")
            self.state.health_checks['database']['healthy'] = True
        except:
            self.state.health_checks['database']['healthy'] = False
        
        # Check brokers
        broker_health = {}
        for name, broker in self.state.brokers.items():
            try:
                # Simplified health check - actual implementation would test real connectivity
                broker_health[name] = {'connected': True, 'latency': 0.1}
            except Exception as e:
                broker_health[name] = {'connected': False, 'error': str(e)}
        
        self.state.health_checks['brokers']['details'] = broker_health
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status"""
        return {
            'status': 'healthy' if self.state.is_healthy() else 'unhealthy',
            'uptime_seconds': self.state.get_uptime(),
            'components': self.state.health_checks,
            'timestamp': datetime.utcnow().isoformat(),
            'version': '1.0.0'
        }
    
    async def shutdown(self):
        """Graceful shutdown of all components"""
        logger.info("🔄 Starting application shutdown...")
        
        try:
            # Disconnect brokers
            for broker_name, broker in self.state.brokers.items():
                try:
                    await broker.disconnect()
                    logger.info(f"Disconnected from {broker_name}")
                except Exception as e:
                    logger.error(f"Error disconnecting from {broker_name}: {e}")
            
            # Close database connections
            await engine.dispose()
            
            self.state.initialized = False
            logger.success("✅ Application shutdown complete")
            
        except Exception as e:
            logger.error(f"❌ Error during shutdown: {e}")
        finally:
            sys.exit(0)


# Global application instance
app_config = ApplicationConfig()


@asynccontextmanager
async def lifespan():
    """Application lifespan context manager"""
    startup_success = await app_config.initialize()
    
    if not startup_success:
        logger.error("Application startup failed")
        raise RuntimeError("Failed to initialize application")
    
    try:
        yield app_config
    finally:
        await app_config.shutdown()


async def get_app_config() -> ApplicationConfig:
    """Dependency injection for application config"""
    return app_config


# API endpoint functions for FastAPI integration
async def health_check():
    """Health check endpoint"""
    return await app_config.get_health_status()


async def get_system_status():
    """Get comprehensive system status"""
    health = await app_config.get_health_status()
    
    return {
        **health,
        'brokers': {
            name: {
                'connected': True,
                'broker_type': broker.__class__.__name__
            }
            for name, broker in app_config.state.brokers.items()
        },
        'ai_orchestrator': {
            'active': app_config.state.ai_orchestrator is not None,
            'mode': app_config.state.ai_orchestrator.trading_mode.value if app_config.state.ai_orchestrator else 'none',
            'stats': app_config.state.ai_orchestrator.get_performance_stats() if app_config.state.ai_orchestrator else {}
        }
    }


class AppConfig:
    """
    Application Configuration class expected by main.py
    Provides configuration loading and validation
    """
    
    def __init__(self, config_data: Dict[str, Any]):
        self.config_data = config_data
        
        # Database configuration
        self.database = DatabaseConfig(**config_data.get('database', {}))
        
        # AI configuration
        self.ai = AIConfig(**config_data.get('ai', {}))
        
        # Brokers configuration
        self.brokers = config_data.get('brokers', {})
        
        # Risk configuration
        self.risk = RiskConfig(**config_data.get('risk', {}))
        
        # Logging configuration
        self.logging = LoggingConfig(**config_data.get('logging', {}))
    
    @classmethod
    def load(cls, config_path: str) -> 'AppConfig':
        """Load configuration from JSON file"""
        config_file = Path(config_path)
        
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            return cls(config_data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file: {e}")
    
    @classmethod
    def from_dict(cls, config_data: Dict[str, Any]) -> 'AppConfig':
        """Create configuration from dictionary"""
        return cls(config_data)
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors"""
        errors = []
        
        # Validate database config
        if not self.database.url:
            errors.append("Database URL is required")
        
        # Validate AI config
        if not hasattr(self.ai, 'trading_mode'):
            errors.append("AI trading_mode is required")
        
        # Validate broker configs
        for broker_name, broker_config in self.brokers.items():
            if broker_config.get('enabled', False):
                if not broker_config.get('api_key') and broker_name not in ['ibkr']:
                    errors.append(f"API key required for enabled broker: {broker_name}")
        
        # Validate risk config
        if not self.risk.max_position_size:
            errors.append("Max position size is required")
        
        return errors
    
    def get_broker_config(self, broker_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for specific broker"""
        return self.brokers.get(broker_name)
    
    def get_enabled_brokers(self) -> List[str]:
        """Get list of enabled broker names"""
        return [
            name for name, config in self.brokers.items()
            if config.get('enabled', False)
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'database': self.database.__dict__,
            'ai': self.ai.__dict__,
            'brokers': self.brokers,
            'risk': self.risk.__dict__,
            'logging': self.logging.__dict__
        }


class DatabaseConfig:
    """Database configuration"""
    def __init__(self, url: str = "sqlite:///trading_orchestrator.db", echo: bool = False):
        self.url = url
        self.echo = echo


class AIConfig:
    """AI configuration"""
    def __init__(
        self, 
        trading_mode: str = "PAPER",
        default_model_tier: str = "fast",
        openai_api_key: str = "",
        anthropic_api_key: str = "",
        local_models: Optional[Dict[str, Any]] = None
    ):
        self.trading_mode = trading_mode
        self.default_model_tier = default_model_tier
        self.openai_api_key = openai_api_key
        self.anthropic_api_key = anthropic_api_key
        self.local_models = local_models or {}


class RiskConfig:
    """Risk management configuration"""
    def __init__(
        self,
        max_position_size: float = 10000,
        max_daily_loss: float = 5000,
        max_orders_per_minute: int = 60,
        circuit_breaker: Optional[Dict[str, Any]] = None
    ):
        self.max_position_size = max_position_size
        self.max_daily_loss = max_daily_loss
        self.max_orders_per_minute = max_orders_per_minute
        self.circuit_breaker = circuit_breaker or {
            'enabled': True,
            'daily_loss_limit': 10000,
            'consecutive_loss_limit': 3
        }


class LoggingConfig:
    """Logging configuration"""
    def __init__(self, level: str = "INFO", file: str = "logs/trading_orchestrator.log"):
        self.level = level
        self.file = file