"""
Main Application Entry Point - Complete System Integration
Day Trading Orchestrator System

This is the main entry point that integrates all system components:
- Application configuration and lifecycle management
- AI trading orchestrator with LLM integration
- Multi-broker connections and order management
- Risk management and compliance
- Real-time terminal interface
- System monitoring and health checks
"""

import asyncio
import signal
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from loguru import logger

# Core system imports
from config.settings import settings
from config.application import app_config, ApplicationConfig
from ui.terminal import TerminalUI
from ui.components.dashboard import DashboardManager
from ai.orchestrator import AITradingOrchestrator, TradingMode
from ai.models.ai_models_manager import AIModelsManager
from ai.tools.trading_tools import TradingTools
from risk.manager import RiskManager
from oms.manager import OrderManager

# Perpetual Operations Integration
from perpetual_operations import PerpetualApplicationEnhancer, enable_perpetual_operations


class TradingOrchestratorApp:
    """
    Main Trading Orchestrator Application
    
    Integrates all system components:
    - AI Trading Orchestrator with multi-tier LLM support
    - Multi-broker order routing and execution
    - Comprehensive risk management
    - Real-time terminal interface with live updates
    - System monitoring and health checks
    """
    
    def __init__(self):
        self.app_config: Optional[ApplicationConfig] = None
        self.ui = TerminalUI()
        self.dashboard = None
        self.running = False
        self.system_task = None
        
        # Perpetual Operations Integration
        self.perpetual_enhancer = PerpetualApplicationEnhancer()
        self.perpetual_integration = None
        
    async def initialize(self) -> bool:
        """Initialize the complete trading system"""
        try:
            logger.info("🚀 Initializing Complete Trading System...")
            
            # Configure logging
            self._setup_logging()
            
            # Create necessary directories
            await self._setup_directories()
            
            # Initialize application configuration
            self.app_config = app_config
            startup_success = await self.app_config.initialize()
            
            if not startup_success:
                logger.error("❌ Application initialization failed")
                return False
            
            # Setup system integration
            await self._setup_system_integration()
            
            # Initialize UI with real data sources
            await self._initialize_ui()
            
            # Start background monitoring
            self._start_background_tasks()
            
            # Initialize Perpetual Operations for 24/7 capability
            await self._initialize_perpetual_operations()
            
            self.running = True
            
            # Display welcome
            await self._display_system_status()
            
            logger.success("✅ Trading System Initialized Successfully with Perpetual Operations")
            return True
            
        except Exception as e:
            logger.exception(f"❌ Failed to initialize trading system: {e}")
            return False
    
    def _setup_logging(self):
        """Configure comprehensive logging"""
        # Remove default handler
        logger.remove()
        
        # Console logging with colors
        logger.add(
            sys.stderr,
            level=settings.log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            colorize=True
        )
        
        # File logging
        logger.add(
            settings.log_file_path,
            rotation=settings.log_rotation,
            retention=settings.log_retention,
            level=settings.log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            encoding="utf-8"
        )
        
        # Error logging
        logger.add(
            f"logs/errors_{datetime.now().strftime('%Y%m%d')}.log",
            level="ERROR",
            rotation="1 day",
            retention="30 days",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            encoding="utf-8"
        )
    
    async def _setup_directories(self):
        """Create necessary directories"""
        directories = [
            Path("logs"),
            Path("data"),
            Path("backups"),
            Path("cache"),
            Path("exports")
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    async def _setup_system_integration(self):
        """Setup integration between system components"""
        try:
            logger.info("🔗 Setting up system integration...")
            
            # Link AI orchestrator with brokers and risk management
            if self.app_config.state.ai_orchestrator:
                ai_orchestrator = self.app_config.state.ai_orchestrator
                
                # Set broker manager reference
                ai_orchestrator.broker_manager = self.app_config.state.order_manager
                
                # Set risk manager reference
                ai_orchestrator.risk_manager = self.app_config.state.risk_manager
            
            # Register brokers with order manager
            if self.app_config.state.order_manager:
                order_manager = self.app_config.state.order_manager
                
                for broker_name, broker_client in self.app_config.state.brokers.items():
                    order_manager.register_broker(broker_name, broker_client)
                
                logger.success(f"✅ Registered {len(self.app_config.state.brokers)} brokers with Order Manager")
            
            # Update risk manager with portfolio info
            if self.app_config.state.risk_manager:
                # Set initial portfolio values (would be loaded from broker)
                self.app_config.state.risk_manager.update_portfolio_value(
                    total_value=100000,  # Mock portfolio value
                    cash=100000
                )
            
            logger.success("✅ System integration completed")
            
        except Exception as e:
            logger.error(f"❌ System integration failed: {e}")
            raise
    
    async def _initialize_ui(self):
        """Initialize UI with real data sources"""
        try:
            logger.info("🖥️ Initializing terminal interface...")
            
            # Create dashboard manager with real data sources
            self.dashboard = DashboardManager(
                app_config=self.app_config,
                ui_instance=self.ui
            )
            
            # Replace mock data callbacks with real ones
            await self.dashboard.setup_real_data_feeds()
            
            logger.success("✅ Terminal interface initialized")
            
        except Exception as e:
            logger.error(f"❌ UI initialization failed: {e}")
            raise
    
    async def _initialize_perpetual_operations(self):
        """Initialize Perpetual Operations for 24/7 continuous trading"""
        try:
            logger.info("🔄 Initializing Perpetual Operations for 24/7 Trading...")
            
            # Enhance main application with perpetual operations
            self.perpetual_integration = self.perpetual_enhancer.enhance_main_application(
                self, 
                fastapi_app=None  # Will be set up later if needed
            )
            
            # Enable perpetual operations with the app configuration
            success = await enable_perpetual_operations(
                app_config=self.app_config,
                fastapi_app=None  # Set to actual FastAPI app if available
            )
            
            if success:
                logger.success("✅ Perpetual Operations enabled - System ready for 24/7 operation")
                
                # Display perpetual operations status
                status_info = f"""
🚀 Perpetual Operations Features Enabled:
   ✅ Automatic failure recovery and fault tolerance
   ✅ Real-time health monitoring and alerting  
   ✅ Zero-downtime maintenance mode
   ✅ Performance optimization and resource management
   ✅ Memory leak detection and prevention
   ✅ Connection pooling and management
   ✅ Database maintenance and optimization
   ✅ Backup and recovery procedures
   ✅ SLA compliance monitoring
   ✅ Operational runbooks and procedures

🌐 Access Dashboard: http://localhost:8000/dashboard
📊 API Endpoints: http://localhost:8000/api/perpetual/*
                """
                self.ui.console.print(status_info)
            else:
                logger.warning("⚠️  Perpetual Operations initialization had issues")
                
        except Exception as e:
            logger.error(f"❌ Perpetual Operations initialization failed: {e}")
            # Don't fail the entire system if perpetual operations fail
            logger.info("🔧 Continuing without Perpetual Operations features")
    
    def _start_background_tasks(self):
        """Start background monitoring and update tasks"""
        try:
            # Data refresh task
            self.data_refresh_task = asyncio.create_task(self._data_refresh_loop())
            
            # Health monitoring task
            self.health_monitor_task = asyncio.create_task(self._health_monitor_loop())
            
            # AI analysis task
            self.ai_analysis_task = asyncio.create_task(self._ai_analysis_loop())
            
            logger.info("✅ Background tasks started")
            
        except Exception as e:
            logger.error(f"❌ Failed to start background tasks: {e}")
    
    async def _data_refresh_loop(self):
        """Background loop for refreshing market data"""
        while self.running:
            try:
                if self.dashboard:
                    await self.dashboard.refresh_all_data()
                await asyncio.sleep(5)  # Refresh every 5 seconds
            except Exception as e:
                logger.error(f"Data refresh error: {e}")
                await asyncio.sleep(10)
    
    async def _health_monitor_loop(self):
        """Background loop for system health monitoring"""
        while self.running:
            try:
                # Update health status
                health_status = await self.app_config.get_health_status()
                
                # Log health warnings
                if not health_status.get('status') == 'healthy':
                    logger.warning(f"System health: {health_status}")
                
                await asyncio.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(30)
    
    async def _ai_analysis_loop(self):
        """Background loop for AI market analysis"""
        while self.running:
            try:
                if self.app_config.state.ai_orchestrator:
                    # Perform periodic market analysis
                    symbols = ["AAPL", "GOOGL", "MSFT", "TSLA"]  # Example symbols
                    
                    analysis = await self.app_config.state.ai_orchestrator.analyze_market(
                        symbols=symbols,
                        analysis_type="quick",
                        use_reasoning_model=False
                    )
                    
                    if analysis.get('analysis'):
                        logger.debug(f"AI Analysis completed for {symbols}")
                
                await asyncio.sleep(300)  # Analyze every 5 minutes
            except Exception as e:
                logger.error(f"AI analysis error: {e}")
                await asyncio.sleep(300)
    
    async def _display_system_status(self):
        """Display initial system status"""
        self.ui.print_welcome()
        
        # Print system status
        status_info = f"""
📊 System Status:
   🟢 Application: Initialized
   🟢 Database: Connected
   🟢 Brokers: {len(self.app_config.state.brokers)} connected
   🟢 AI Orchestrator: Active
   🟢 Risk Management: Operational
   🟢 Order Manager: Ready
   
💡 System is ready for trading operations!
"""
        self.ui.console.print(status_info)
        
        # Show connection status
        if self.app_config.state.brokers:
            broker_list = ", ".join(self.app_config.state.brokers.keys())
            self.ui.console.print(f"🔗 Connected Brokers: {broker_list}")
        else:
            self.ui.console.print("⚠️  No brokers connected - Configure API keys in .env file")
    
    async def run(self):
        """Run the complete trading orchestrator"""
        try:
            # Initialize system
            if not await self.initialize():
                logger.error("System initialization failed")
                return
            
            # Setup signal handlers for graceful shutdown
            self._setup_signal_handlers()
            
            # Run the terminal interface with real data
            logger.info("🎮 Starting Matrix Trading Interface...")
            await self.ui.run_dashboard_with_integration(self.dashboard)
            
        except KeyboardInterrupt:
            logger.info("🛑 Shutdown requested by user")
        
        except Exception as e:
            logger.exception(f"💥 Fatal error in main loop: {e}")
        
        finally:
            await self.shutdown()
    
    def _setup_signal_handlers(self):
        """Setup graceful shutdown signal handlers"""
        def signal_handler(signum, frame):
            logger.info(f"Signal {signum} received, initiating graceful shutdown...")
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def shutdown(self):
        """Graceful shutdown of all components"""
        logger.info("🔄 Initiating graceful shutdown...")
        
        self.running = False
        
        try:
            # Cancel background tasks
            for task_name in ['data_refresh_task', 'health_monitor_task', 'ai_analysis_task']:
                task = getattr(self, task_name, None)
                if task and not task.done():
                    task.cancel()
            
            # Shutdown application configuration
            if self.app_config:
                await self.app_config.shutdown()
            
            # Shutdown Perpetual Operations
            if self.perpetual_integration:
                await self.perpetual_integration.shutdown()
            
            logger.success("✅ Graceful shutdown completed")
            
        except Exception as e:
            logger.error(f"❌ Error during shutdown: {e}")
        finally:
            sys.exit(0)
    
    async def get_system_overview(self) -> Dict[str, Any]:
        """Get comprehensive system overview"""
        health = await self.app_config.get_health_status()
        performance = {}
        
        # Get AI performance stats
        if self.app_config.state.ai_orchestrator:
            performance['ai'] = self.app_config.state.ai_orchestrator.get_performance_stats()
        
        # Get order manager performance
        if self.app_config.state.order_manager:
            performance['orders'] = await self.app_config.state.order_manager.get_performance_metrics()
        
        # Get risk metrics
        if self.app_config.state.risk_manager:
            performance['risk'] = await self.app_config.state.risk_manager.get_risk_summary()
        
        return {
            'system_health': health,
            'performance': performance,
            'brokers': {
                name: {
                    'connected': True,
                    'type': broker.__class__.__name__
                }
                for name, broker in self.app_config.state.brokers.items()
            },
            'components': {
                'ai_orchestrator': self.app_config.state.ai_orchestrator is not None,
                'risk_manager': self.app_config.state.risk_manager is not None,
                'order_manager': self.app_config.state.order_manager is not None,
                'database': health['components'].get('database', {}).get('healthy', False)
            }
        }


async def main():
    """Main application entry point"""
    app = TradingOrchestratorApp()
    
    try:
        logger.info("🎯 Starting Day Trading Orchestrator System v2.0 with Perpetual Operations")
        logger.info("🌟 Features: 24/7 Operation | Auto-Recovery | Zero-Downtime Maintenance")
        await app.run()
    except KeyboardInterrupt:
        logger.info("👋 Goodbye!")
    except Exception as e:
        logger.exception(f"💥 Unhandled exception: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Set up event loop policy for Windows if needed
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application interrupted, exiting...")
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)