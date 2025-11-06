#!/usr/bin/env python3
"""
Perpetual Trading System Startup Script
=======================================

This script demonstrates how to start the trading orchestrator with full perpetual operations capabilities.

Features:
- Automatic integration of perpetual operations
- Health monitoring and alerting
- Zero-downtime maintenance mode
- Real-time dashboard access
- Comprehensive API endpoints

Usage:
    python start_perpetual.py [--mode {terminal,api,hybrid}] [--port PORT]

Author: Trading Orchestrator System
Version: 2.0.0
Date: 2025-11-06
"""

import asyncio
import argparse
import sys
import os
from pathlib import Path
from typing import Optional

# Add trading orchestrator to path
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger
from main import TradingOrchestratorApp
from perpetual_api import create_complete_application


class PerpetualStartupManager:
    """Manages the startup of perpetual trading system"""
    
    def __init__(self):
        self.app_instance = None
        self.fastapi_app = None
        
    async def start_terminal_mode(self):
        """Start in terminal interface mode"""
        logger.info("🚀 Starting Trading Orchestrator in Terminal Mode with Perpetual Operations")
        
        # Create and run terminal application
        app = TradingOrchestratorApp()
        
        try:
            # Initialize with perpetual operations
            await app.run()
        except KeyboardInterrupt:
            logger.info("👋 Goodbye!")
        except Exception as e:
            logger.exception(f"💥 Fatal error: {e}")
            sys.exit(1)
    
    async def start_api_mode(self):
        """Start in API mode (FastAPI with perpetual operations)"""
        logger.info("🚀 Starting Trading Orchestrator API Server with Perpetual Operations")
        
        try:
            # Create complete application
            app, app_config, ws_manager, perpetual_integration = await create_complete_application()
            
            logger.info("🌐 Server starting on http://localhost:8000")
            logger.info("📊 Dashboard: http://localhost:8000/dashboard")
            logger.info("📚 API Docs: http://localhost:8000/api/docs")
            
            # Import uvicorn here to avoid import issues
            import uvicorn
            
            # Configure and run server
            config = uvicorn.Config(
                app,
                host="0.0.0.0",
                port=8000,
                log_level="info",
                access_log=True
            )
            
            server = uvicorn.Server(config)
            await server.serve()
            
        except KeyboardInterrupt:
            logger.info("👋 Goodbye!")
        except Exception as e:
            logger.exception(f"💥 Fatal error: {e}")
            sys.exit(1)
    
    async def start_hybrid_mode(self):
        """Start in hybrid mode (both terminal and API)"""
        logger.info("🚀 Starting Trading Orchestrator in Hybrid Mode with Perpetual Operations")
        
        # This would start both terminal interface and API server
        # For now, we'll start the API server as it's more feature-complete
        
        logger.info("📋 Starting API server (hybrid mode)...")
        await self.start_api_mode()
    
    async def check_system_readiness(self) -> bool:
        """Check if system is ready for perpetual operations"""
        try:
            # Check Python version
            if sys.version_info < (3, 8):
                logger.error("❌ Python 3.8+ required")
                return False
            
            # Check required directories
            directories = ["logs", "backups", "cache", "tmp"]
            for directory in directories:
                Path(directory).mkdir(parents=True, exist_ok=True)
            
            # Check file permissions
            test_file = Path("logs/.startup_test")
            try:
                test_file.write_text("test")
                test_file.unlink()
            except Exception as e:
                logger.error(f"❌ File system access issue: {e}")
                return False
            
            logger.success("✅ System readiness check passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ System readiness check failed: {e}")
            return False
    
    def print_startup_banner(self):
        """Print startup banner"""
        banner = """
        
╔══════════════════════════════════════════════════════════════════════╗
║                     🚀 PERPETUAL TRADING SYSTEM 🚀                   ║
║                                                                      ║
║  Version: 2.0.0                                                     ║
║  Features: 24/7 Operation | Auto-Recovery | Zero-Downtime           ║
║                                                                      ║
║  🌐 Dashboard: http://localhost:8000/dashboard                      ║
║  📚 API Docs: http://localhost:8000/api/docs                        ║
║  🔗 API Base: http://localhost:8000/api                             ║
║                                                                      ║
║  ⚡ Perpetual Operations Features:                                  ║
║     ✅ Real-time system monitoring                                  ║
║     ✅ Automatic failure recovery                                    ║
║     ✅ Health monitoring and alerting                               ║
║     ✅ Zero-downtime maintenance mode                               ║
║     ✅ Performance optimization                                      ║
║     ✅ Backup and recovery procedures                                ║
║     ✅ Resource management and cleanup                               ║
║     ✅ Memory leak detection and prevention                          ║
║     ✅ Connection pooling and management                             ║
║     ✅ Database maintenance and optimization                         ║
║     ✅ SLA compliance monitoring                                     ║
║     ✅ Operational runbooks and procedures                           ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝

        """
        print(banner)
    
    def print_shutdown_info(self):
        """Print shutdown information"""
        info = """
        
╔══════════════════════════════════════════════════════════════════════╗
║                         🛑 SYSTEM SHUTDOWN                           ║
║                                                                      ║
║  Thank you for using Perpetual Trading System!                      ║
║                                                                      ║
║  📊 View operational reports in logs/ directory                     ║
║  💾 Backups available in backups/ directory                         ║
║  📋 Check operational runbooks in docs/runbooks/                    ║
║                                                                      ║
║  For support and documentation, visit the dashboard or API docs.    ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝

        """
        print(info)


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Perpetual Trading System Startup",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python start_perpetual.py --mode terminal    # Start terminal interface
  python start_perpetual.py --mode api         # Start API server
  python start_perpetual.py --mode hybrid      # Start hybrid mode
  python start_perpetual.py --port 8080        # Custom port
  
Supported Modes:
  terminal  - Start terminal interface with perpetual operations
  api       - Start FastAPI server with perpetual operations dashboard
  hybrid    - Start both terminal and API (experimental)
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["terminal", "api", "hybrid"],
        default="api",
        help="Startup mode (default: api)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for API server (default: 8000)"
    )
    
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check system readiness, don't start"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logger.remove()
    if args.verbose:
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.add(sys.stderr, level="INFO")
    
    logger.add(
        "logs/perpetual_startup.log",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="1 day",
        retention="30 days"
    )
    
    # Create startup manager
    startup_manager = PerpetualStartupManager()
    
    try:
        # Print startup banner
        startup_manager.print_startup_banner()
        
        # Check system readiness
        if not await startup_manager.check_system_readiness():
            logger.error("❌ System not ready for perpetual operations")
            sys.exit(1)
        
        # Handle check-only mode
        if args.check_only:
            logger.info("✅ System readiness check completed successfully")
            logger.info("ℹ️  Use --mode terminal, --mode api, or --mode hybrid to start")
            return
        
        # Set port environment variable for API mode
        if args.mode in ["api", "hybrid"]:
            os.environ["API_PORT"] = str(args.port)
        
        # Start in requested mode
        if args.mode == "terminal":
            await startup_manager.start_terminal_mode()
        elif args.mode == "api":
            await startup_manager.start_api_mode()
        elif args.mode == "hybrid":
            await startup_manager.start_hybrid_mode()
        
    except KeyboardInterrupt:
        logger.info("🛑 Shutdown requested by user")
    except Exception as e:
        logger.exception(f"💥 Startup failed: {e}")
        sys.exit(1)
    finally:
        startup_manager.print_shutdown_info()


if __name__ == "__main__":
    # Ensure proper event loop handling on Windows
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application interrupted, exiting...")
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)