#!/usr/bin/env python3
"""
Day Trading Orchestrator - Main Application Entry Point
Matrix-Themed Terminal Interface with Multi-Broker Integration
"""

import asyncio
import logging
import sys
import signal
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config.application import AppConfig
from database.models import init_database
from brokers.factory import BrokerFactory
from ai.orchestrator import AITradingOrchestrator
from ai.models.ai_models_manager import AIModelsManager
from ai.tools.trading_tools import TradingTools
from ui.interface import MatrixTerminalInterface
from risk.risk_engine import RiskEngine
from oms.oms_engine import OMSEngine
from strategies.registry import StrategyRegistry
from market_data.market_data_manager import MarketDataManager
from utils.logger import setup_logging

class TradingOrchestratorApp:
    """Main application orchestrator for the day trading system."""
    
    def __init__(self):
        self.config = None
        self.database = None
        self.broker_factory = None
        self.ai_models = None
        self.trading_tools = None
        self.ai_orchestrator = None
        self.risk_engine = None
        self.oms_engine = None
        self.strategy_registry = None
        self.market_data_manager = None
        self.terminal_interface = None
        self.logger = None
        self.running = False
    
    async def initialize(self, config_path: str = "config.json"):
        """Initialize all system components."""
        try:
            # Setup logging
            self.logger = setup_logging()
            self.logger.info("🚀 Initializing Day Trading Orchestrator...")
            
            # Load configuration
            self.config = AppConfig.load(config_path)
            self.logger.info(f"📋 Configuration loaded from {config_path}")
            
            # Initialize database
            await init_database(self.config.database)
            self.logger.info("💾 Database initialized")
            
            # Initialize broker factory
            self.broker_factory = BrokerFactory()
            await self.broker_factory.initialize(self.config.brokers)
            self.logger.info(f"🔌 Broker factory initialized with {len(self.broker_factory.get_available_brokers())} brokers")
            
            # Initialize AI models
            self.ai_models = AIModelsManager(
                openai_api_key=self.config.ai.openai_api_key,
                anthropic_api_key=self.config.ai.anthropic_api_key,
                local_model_config=self.config.ai.local_models
            )
            self.logger.info("🤖 AI models initialized")
            
            # Initialize trading tools
            self.trading_tools = TradingTools(self.broker_factory)
            self.logger.info("🛠️ Trading tools initialized")
            
            # Initialize AI orchestrator
            self.ai_orchestrator = AITradingOrchestrator(
                ai_models_manager=self.ai_models,
                trading_tools=self.trading_tools,
                trading_mode=self.config.ai.trading_mode
            )
            self.logger.info("🧠 AI orchestrator initialized")
            
            # Initialize risk engine
            self.risk_engine = RiskEngine(self.config.risk)
            await self.risk_engine.initialize()
            self.logger.info("⚡ Risk engine initialized")
            
            # Initialize OMS
            self.oms_engine = OMSEngine(self.broker_factory, self.risk_engine)
            await self.oms_engine.initialize()
            self.logger.info("📋 OMS initialized")
            
            # Initialize strategy registry
            self.strategy_registry = StrategyRegistry()
            await self.strategy_registry.initialize()
            self.logger.info("📈 Strategy registry initialized")
            
            # Initialize market data manager
            self.market_data_manager = MarketDataManager(self.broker_factory)
            await self.market_data_manager.initialize()
            self.logger.info("📊 Market data manager initialized")
            
            # Initialize terminal interface
            self.terminal_interface = MatrixTerminalInterface(
                ai_orchestrator=self.ai_orchestrator,
                broker_factory=self.broker_factory,
                risk_engine=self.risk_engine,
                oms_engine=self.oms_engine,
                strategy_registry=self.strategy_registry,
                market_data_manager=self.market_data_manager,
                config=self.config
            )
            self.logger.info("💻 Matrix terminal interface initialized")
            
            self.running = True
            self.logger.info("✅ System initialization complete!")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize system: {str(e)}")
            raise
    
    async def start(self):
        """Start the trading orchestrator."""
        if not self.running:
            await self.initialize()
        
        self.logger.info("🏁 Starting Day Trading Orchestrator...")
        
        try:
            # Start terminal interface
            await self.terminal_interface.start()
            
        except KeyboardInterrupt:
            self.logger.info("🛑 Received interrupt signal")
        except Exception as e:
            self.logger.error(f"❌ Application error: {str(e)}")
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """Gracefully shutdown the system."""
        self.logger.info("🔄 Shutting down system...")
        self.running = False
        
        try:
            # Shutdown components in reverse order
            if self.terminal_interface:
                await self.terminal_interface.shutdown()
            
            if self.market_data_manager:
                await self.market_data_manager.shutdown()
            
            if self.strategy_registry:
                await self.strategy_registry.shutdown()
            
            if self.oms_engine:
                await self.oms_engine.shutdown()
            
            if self.risk_engine:
                await self.risk_engine.shutdown()
            
            if self.ai_orchestrator:
                await self.ai_orchestrator.shutdown()
            
            if self.broker_factory:
                await self.broker_factory.shutdown()
            
            if self.ai_models:
                await self.ai_models.shutdown()
            
            self.logger.info("✅ System shutdown complete")
            
        except Exception as e:
            self.logger.error(f"❌ Error during shutdown: {str(e)}")

async def quick_start(config_path: str = "config.json"):
    """Quick start function for programmatic usage."""
    app = TradingOrchestratorApp()
    await app.initialize(config_path)
    return app

def create_default_config():
    """Create a default configuration file."""
    config = {
        "database": {
            "url": "sqlite:///trading_orchestrator.db",
            "echo": False
        },
        "ai": {
            "trading_mode": "PAPER",
            "default_model_tier": "fast"
        },
        "brokers": {
            "alpaca": {
                "enabled": True,
                "api_key": "YOUR_ALPACA_API_KEY",
                "secret_key": "YOUR_ALPACA_SECRET_KEY",
                "paper": True
            },
            "binance": {
                "enabled": True,
                "api_key": "YOUR_BINANCE_API_KEY",
                "secret_key": "YOUR_BINANCE_SECRET_KEY",
                "testnet": True
            },
            "ibkr": {
                "enabled": False,
                "host": "127.0.0.1",
                "port": 7497,
                "paper": True
            }
        },
        "risk": {
            "max_position_size": 10000,
            "max_daily_loss": 5000,
            "max_orders_per_minute": 60,
            "circuit_breaker": {
                "enabled": True,
                "daily_loss_limit": 10000,
                "consecutive_loss_limit": 3
            }
        },
        "logging": {
            "level": "INFO",
            "file": "logs/trading_orchestrator.log"
        }
    }
    
    import json
    with open("config.json", "w") as f:
        json.dump(config, f, indent=4)
    
    print("📝 Default configuration created: config.json")
    print("⚠️  Please update with your actual API keys before running!")

def print_banner():
    """Print the Matrix-style banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║    ██████╗ ██╗   ██╗██████╗ ███╗   ███╗                    ║
    ║   ██╔═══██╗██║   ██║██╔══██╗████╗ ████║                    ║
    ║   ██║   ██║██║   ██║██████╔╝██╔████╔██║                    ║
    ║   ██║   ██║██║   ██║██╔══██╗██║╚██╔╝██║                    ║
    ║   ╚██████╔╝╚██████╔╝██████╔╝██║ ╚═╝ ██║                    ║
    ║    ╚═════╝  ╚═════╝ ╚═════╝ ╚═╝     ╚═╝                    ║
    ║                                                              ║
    ║           DAY TRADING ORCHESTRATOR SYSTEM                    ║
    ║                 Multi-Broker AI Trading                     ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    
    ⚡ Matrix-Themed Terminal Interface
    🤖 AI-Powered Trading Decisions
    🔌 7 Broker Integration Support
    🛡️ Enterprise Risk Management
    📈 Multi-Strategy Trading Framework
    """
    print(banner)

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Day Trading Orchestrator")
    parser.add_argument("--config", default="config.json", help="Configuration file path")
    parser.add_argument("--create-config", action="store_true", help="Create default configuration")
    parser.add_argument("--demo", action="store_true", help="Run demo mode")
    parser.add_argument("--version", action="store_true", help="Show version")
    
    args = parser.parse_args()
    
    if args.version:
        print("Day Trading Orchestrator v1.0.0")
        return
    
    if args.create_config:
        create_default_config()
        return
    
    print_banner()
    
    if args.demo:
        print("🎮 Running demo mode...")
        import ui.demo
        ui.demo.run_demo()
        return
    
    try:
        # Setup signal handlers
        def signal_handler(sig, frame):
            print("\n🛑 Shutting down gracefully...")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Start the application
        app = TradingOrchestratorApp()
        asyncio.run(app.start())
        
    except FileNotFoundError:
        print("❌ Configuration file not found!")
        print("💡 Run with --create-config to generate a default configuration")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()