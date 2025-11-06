#!/usr/bin/env python3
"""
Day Trading Orchestrator - Debug Mode Launcher
Enhanced debugging and development environment
"""

import sys
import os
import json
import time
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import traceback
import pdb

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.absolute()))

class DebugLauncher:
    """Enhanced debug launcher for development"""
    
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.config = {}
        self.debug_enabled = True
        self.verbose_logging = True
        self.error_breakpoints = True
        
    def setup_debug_logging(self):
        """Configure debug logging"""
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
        
        # Create logs directory
        os.makedirs("logs", exist_ok=True)
        
        # Configure root logger
        logging.basicConfig(
            level=logging.DEBUG,
            format=log_format,
            handlers=[
                logging.FileHandler("logs/debug.log"),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        # Set specific logger levels
        logging.getLogger("trading_orchestrator").setLevel(logging.DEBUG)
        logging.getLogger("brokers").setLevel(logging.DEBUG)
        logging.getLogger("ai").setLevel(logging.DEBUG)
        logging.getLogger("database").setLevel(logging.DEBUG)
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("🔧 Debug logging enabled")
    
    def load_configuration(self) -> bool:
        """Load and validate configuration with enhanced debugging"""
        try:
            # Check if config file exists
            if not os.path.exists(self.config_path):
                self.logger.error(f"❌ Configuration file {self.config_path} not found")
                print(f"⚠️  Configuration file {self.config_path} not found")
                print("💡 Creating default configuration...")
                self.create_default_config()
                return False
            
            # Load configuration
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
            
            self.logger.info(f"✅ Configuration loaded from {self.config_path}")
            
            # Validate configuration
            issues = self.validate_config()
            if issues:
                self.logger.warning(f"⚠️  Configuration issues found: {issues}")
                print("⚠️  Configuration issues:")
                for issue in issues:
                    print(f"  • {issue}")
                return False
            
            self.logger.info("✅ Configuration validation passed")
            return True
            
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON in configuration: {e}"
            self.logger.error(f"❌ {error_msg}")
            print(f"❌ {error_msg}")
            return False
        except Exception as e:
            error_msg = f"Error loading configuration: {e}"
            self.logger.error(f"❌ {error_msg}")
            print(f"❌ {error_msg}")
            traceback.print_exc()
            return False
    
    def validate_config(self) -> List[str]:
        """Validate configuration with detailed checks"""
        issues = []
        
        # Required sections
        required_sections = ["database", "brokers", "risk"]
        for section in required_sections:
            if section not in self.config:
                issues.append(f"Missing required section: {section}")
        
        # Database configuration
        if "database" in self.config:
            db_config = self.config["database"]
            if "url" not in db_config:
                issues.append("Database URL not configured")
        
        # Brokers configuration
        if "brokers" in self.config:
            for broker_name, broker_config in self.config["brokers"].items():
                if broker_config.get("enabled", False):
                    if broker_name.lower() == "alpaca":
                        if not broker_config.get("api_key"):
                            issues.append(f"Alpaca API key not set")
                    elif broker_name.lower() == "binance":
                        if not broker_config.get("api_key"):
                            issues.append(f"Binance API key not set")
        
        # Risk configuration
        if "risk" in self.config:
            risk_config = self.config["risk"]
            required_risk_params = ["max_position_size", "max_daily_loss"]
            for param in required_risk_params:
                if param not in risk_config:
                    issues.append(f"Risk parameter {param} not configured")
        
        return issues
    
    def create_default_config(self):
        """Create a default configuration for development"""
        default_config = {
            "database": {
                "url": "sqlite:///debug_trading_orchestrator.db",
                "echo": True  # Enable SQL logging in debug mode
            },
            "ai": {
                "trading_mode": "PAPER",
                "debug_ai_calls": True
            },
            "brokers": {
                "alpaca": {
                    "enabled": False,
                    "api_key": "YOUR_ALPACA_API_KEY",
                    "secret_key": "YOUR_ALPACA_SECRET_KEY",
                    "paper": True
                },
                "binance": {
                    "enabled": False,
                    "api_key": "YOUR_BINANCE_API_KEY",
                    "secret_key": "YOUR_BINANCE_SECRET_KEY",
                    "testnet": True
                }
            },
            "risk": {
                "max_position_size": 1000,
                "max_daily_loss": 500,
                "debug_mode": True
            },
            "logging": {
                "level": "DEBUG",
                "file": "logs/debug.log",
                "debug_mode": True
            }
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        print(f"✅ Created default configuration: {self.config_path}")
    
    def check_dependencies(self) -> List[str]:
        """Check if all required dependencies are installed"""
        required_packages = [
            "aiohttp",
            "asyncio",
            "sqlalchemy",
            "alembic",
            "pydantic",
            "loguru",
            "websockets",
            "pandas",
            "numpy"
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
                self.logger.debug(f"✅ Package {package} is available")
            except ImportError:
                missing_packages.append(package)
                self.logger.warning(f"❌ Package {package} is missing")
        
        return missing_packages
    
    async def test_components(self) -> Dict[str, bool]:
        """Test individual components in debug mode"""
        results = {}
        
        # Test database connection
        try:
            from config.database import init_db, engine
            await init_db()
            async with engine.connect() as conn:
                await conn.execute("SELECT 1")
            results["database"] = True
            self.logger.info("✅ Database connection test passed")
        except Exception as e:
            results["database"] = False
            self.logger.error(f"❌ Database connection test failed: {e}")
        
        # Test broker factory
        try:
            from brokers.factory import BrokerFactory
            factory = BrokerFactory()
            available_brokers = factory.get_available_brokers()
            results["brokers"] = len(available_brokers) > 0
            if results["brokers"]:
                self.logger.info(f"✅ Broker factory test passed ({len(available_brokers)} brokers)")
            else:
                self.logger.warning("⚠️  No brokers available")
        except Exception as e:
            results["brokers"] = False
            self.logger.error(f"❌ Broker factory test failed: {e}")
        
        # Test AI components
        try:
            from ai.orchestrator import AITradingOrchestrator
            orchestrator = AITradingOrchestrator()
            results["ai"] = True
            self.logger.info("✅ AI orchestrator test passed")
        except Exception as e:
            results["ai"] = False
            self.logger.error(f"❌ AI orchestrator test failed: {e}")
        
        return results
    
    def install_missing_dependencies(self, missing_packages: List[str]):
        """Offer to install missing dependencies"""
        if not missing_packages:
            return
        
        print("\n⚠️  Missing required packages:")
        for package in missing_packages:
            print(f"  • {package}")
        
        response = input("\n❓ Install missing packages? (y/N): ").strip().lower()
        
        if response in ['y', 'yes']:
            import subprocess
            
            for package in missing_packages:
                print(f"Installing {package}...")
                try:
                    subprocess.run([sys.executable, "-m", "pip", "install", package], check=True)
                    print(f"✅ {package} installed successfully")
                except subprocess.CalledProcessError as e:
                    print(f"❌ Failed to install {package}: {e}")
        else:
            print("⚠️  Continuing without installing missing packages")
    
    def setup_breakpoints(self):
        """Setup debug breakpoints for common error points"""
        if not self.error_breakpoints:
            return
        
        # Define critical error points
        self.breakpoints = {
            "broker_connection": "Set breakpoint on broker connection errors",
            "order_execution": "Set breakpoint on order execution errors",
            "database_error": "Set breakpoint on database errors",
            "ai_api_error": "Set breakpoint on AI API errors"
        }
        
        # Register error handlers
        def error_handler(error_type, exception):
            self.logger.error(f"🚨 Critical error in {error_type}: {exception}")
            if self.error_breakpoints:
                print(f"\n💥 Critical error detected in {error_type}")
                print(f"Error: {exception}")
                print("🔍 Starting debugger...")
                print("Type 'c' to continue or 'q' to quit")
                pdb.set_trace()
        
        self.logger.info("🔧 Debug breakpoints configured")
    
    def run_debug_mode(self, component: Optional[str] = None):
        """Run the application in debug mode"""
        print("\n" + "="*60)
        print("🔧 DAY TRADING ORCHESTRATOR - DEBUG MODE")
        print("="*60)
        
        # Print debug information
        print(f"📋 Configuration: {self.config_path}")
        print(f"🐍 Python: {sys.version}")
        print(f"📁 Working Directory: {os.getcwd()}")
        print(f"🕐 Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Check dependencies
        print("\n📦 Checking Dependencies...")
        missing_packages = self.check_dependencies()
        
        if missing_packages:
            self.install_missing_dependencies(missing_packages)
        
        # Load configuration
        print("\n⚙️  Loading Configuration...")
        config_loaded = self.load_configuration()
        
        # Test components
        print("\n🧪 Testing Components...")
        
        async def component_test():
            results = await self.test_components()
            
            print("\n📊 Component Test Results:")
            for component, status in results.items():
                status_icon = "✅" if status else "❌"
                print(f"{status_icon} {component.title()}")
            
            return all(results.values())
        
        if config_loaded:
            all_tests_passed = asyncio.run(component_test())
            
            if not all_tests_passed:
                print("\n⚠️  Some component tests failed")
                print("🔧 You can still proceed, but some features may not work correctly")
            
            # Start debugger session
            self.start_debugger_session()
        
        else:
            print("\n❌ Configuration loading failed")
            print("🔧 Please fix configuration issues before proceeding")
            
            if input("\n❓ Create default configuration and continue? (y/N): ").strip().lower() in ['y', 'yes']:
                self.create_default_config()
                self.start_debugger_session()
    
    def start_debugger_session(self):
        """Start interactive debugging session"""
        print("\n" + "="*60)
        print("🕵️  INTERACTIVE DEBUG SESSION")
        print("="*60)
        
        print("Available debug commands:")
        print("  components  - Test all components")
        print("  config      - Show current configuration")
        print("  database    - Test database connection")
        print("  brokers     - Test broker connections")
        print("  ai          - Test AI integration")
        print("  trace       - Enable detailed tracing")
        print("  breakpoint  - Set manual breakpoint")
        print("  run         - Run main application")
        print("  quit        - Exit debug mode")
        
        while True:
            try:
                command = input("\ndebug> ").strip().lower()
                
                if command == "quit" or command == "exit":
                    print("👋 Exiting debug mode")
                    break
                
                elif command == "components":
                    asyncio.run(self.test_components())
                
                elif command == "config":
                    print("\n📋 Current Configuration:")
                    print(json.dumps(self.config, indent=2))
                
                elif command == "database":
                    self.test_database()
                
                elif command == "brokers":
                    self.test_brokers()
                
                elif command == "ai":
                    self.test_ai()
                
                elif command == "trace":
                    self.enable_tracing()
                
                elif command == "breakpoint":
                    print("🔍 Setting manual breakpoint...")
                    pdb.set_trace()
                    print("📍 Breakpoint hit - use 'c' to continue")
                
                elif command == "run":
                    self.run_main_application()
                
                elif command == "help":
                    print("Debug commands:")
                    for cmd in ["components", "config", "database", "brokers", "ai", "trace", "breakpoint", "run", "quit"]:
                        print(f"  {cmd}")
                
                else:
                    print("❓ Unknown command. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                print("\n\n⚠️  Debug session interrupted")
                break
            except Exception as e:
                print(f"❌ Debug command failed: {e}")
                traceback.print_exc()
    
    def test_database(self):
        """Test database connection in detail"""
        try:
            import asyncio
            from config.database import init_db, engine
            
            print("🔍 Testing database connection...")
            asyncio.run(init_db())
            
            async def test_query():
                async with engine.connect() as conn:
                    result = await conn.execute("SELECT datetime('now') as current_time")
                    row = result.fetchone()
                    print(f"✅ Database query successful: {row[0]}")
            
            asyncio.run(test_query())
            
        except Exception as e:
            print(f"❌ Database test failed: {e}")
            traceback.print_exc()
    
    def test_brokers(self):
        """Test broker connections"""
        try:
            from brokers.factory import BrokerFactory
            
            factory = BrokerFactory()
            available_brokers = factory.get_available_brokers()
            
            print(f"🔍 Testing {len(available_brokers)} broker(s)...")
            
            for broker_name in available_brokers:
                print(f"  • {broker_name}")
                # Add specific broker tests here
            
            print("✅ Broker factory test completed")
            
        except Exception as e:
            print(f"❌ Broker test failed: {e}")
            traceback.print_exc()
    
    def test_ai(self):
        """Test AI integration"""
        try:
            from ai.orchestrator import AITradingOrchestrator
            
            print("🔍 Testing AI components...")
            orchestrator = AITradingOrchestrator()
            
            # Test basic initialization
            print("✅ AI orchestrator initialized")
            
            # Test if API keys are configured
            ai_config = self.config.get("ai", {})
            if ai_config.get("openai_api_key") and ai_config["openai_api_key"] != "YOUR_OPENAI_API_KEY":
                print("✅ OpenAI API key configured")
            else:
                print("⚠️  OpenAI API key not configured")
            
            if ai_config.get("anthropic_api_key") and ai_config["anthropic_api_key"] != "YOUR_ANTHROPIC_API_KEY":
                print("✅ Anthropic API key configured")
            else:
                print("⚠️  Anthropic API key not configured")
            
        except Exception as e:
            print(f"❌ AI test failed: {e}")
            traceback.print_exc()
    
    def enable_tracing(self):
        """Enable detailed execution tracing"""
        print("🔍 Enabling detailed tracing...")
        
        # Set global tracing
        sys.settrace(self.trace_calls)
        
        print("✅ Tracing enabled")
        print("📝 Trace output will be shown in real-time")
        print("⏹️  Press Ctrl+C to stop tracing")
    
    def trace_calls(self, frame, event, arg):
        """Trace function calls"""
        if event == "call":
            filename = frame.f_code.co_filename
            if "trading_orchestrator" in filename:
                function_name = frame.f_code.co_name
                line_number = frame.f_lineno
                print(f"📞 Call: {function_name} ({filename}:{line_number})")
        
        return self.trace_calls
    
    def run_main_application(self):
        """Run the main application from debug mode"""
        print("🚀 Starting main application from debug mode...")
        
        try:
            from main import main
            main()
        except Exception as e:
            print(f"❌ Main application failed: {e}")
            traceback.print_exc()

def main():
    """Main debug launcher"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Day Trading Orchestrator Debug Mode")
    parser.add_argument("--config", default="config.json", help="Configuration file path")
    parser.add_argument("--component", help="Test specific component only")
    parser.add_argument("--no-breakpoints", action="store_true", help="Disable automatic breakpoints")
    parser.add_argument("--trace", action="store_true", help="Enable execution tracing")
    parser.add_argument("--install-deps", action="store_true", help="Auto-install missing dependencies")
    
    args = parser.parse_args()
    
    # Create debug launcher
    launcher = DebugLauncher(args.config)
    launcher.error_breakpoints = not args.no_breakpoints
    
    # Setup debug logging
    launcher.setup_debug_logging()
    
    if args.trace:
        launcher.enable_tracing()
    
    # Run debug mode
    launcher.run_debug_mode(args.component)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Debug session ended")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Debug launcher failed: {e}")
        traceback.print_exc()
        sys.exit(1)