#!/usr/bin/env python3
"""
Day Trading Orchestrator Startup Script
Provides easy system startup with configuration checks and diagnostics
"""

import asyncio
import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger
from config.settings import settings
from validate_system import SystemValidator
from main import TradingOrchestratorApp


class SystemLauncher:
    """System launcher with validation and startup"""
    
    def __init__(self):
        self.app = None
    
    async def run(self, validate_only: bool = False, skip_validation: bool = False):
        """Run the trading system"""
        try:
            print("🚀 Day Trading Orchestrator System")
            print("=" * 50)
            
            # Step 1: Validate system
            if not skip_validation:
                print("\n🔍 Step 1: System Validation")
                validator = SystemValidator()
                validation_success = await validator.run_full_validation()
                
                if not validation_success:
                    print("\n💥 System validation failed!")
                    print("🔧 Please fix the errors and try again.")
                    print("💡 Run 'python start_system.py --skip-validation' to bypass checks.")
                    return False
                
                print("\n✅ System validation passed!")
            else:
                print("\n⏭️  Skipping system validation")
            
            # Step 2: Launch application
            if not validate_only:
                print("\n🎮 Step 2: Launching Application")
                
                self.app = TradingOrchestratorApp()
                
                print("💫 Starting the Matrix Trading Interface...")
                print("📝 Press Ctrl+C to exit gracefully")
                print("-" * 50)
                
                await self.app.run()
            else:
                print("\n✅ Validation-only mode complete!")
            
            return True
            
        except KeyboardInterrupt:
            print("\n\n🛑 Shutdown requested by user")
            return True
        except Exception as e:
            print(f"\n💥 System startup failed: {e}")
            logger.exception(f"Startup error: {e}")
            return False
        finally:
            if self.app:
                await self.app.shutdown()


def print_banner():
    """Print startup banner"""
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║   ████████╗██████╗  █████╗ ██████╗ ██╗███╗   ██╗ ██████╗    ║
    ║   ╚══██╔══╝██╔══██╗██╔══██╗██╔══██╗██║████╗  ██║██╔════╝    ║
    ║      ██║   ██████╔╝███████║██║  ██║██║██╔██╗ ██║██║  ███╗   ║
    ║      ██║   ██╔══██╗██╔══██║██║  ██║██║██║╚██╗██║██║   ██║   ║
    ║      ██║   ██║  ██║██║  ██║██████╔╝██║██║ ╚████║╚██████╔╝   ║
    ║      ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ ╚═╝╚═╝  ╚═══╝ ╚═════╝    ║
    ║                                                               ║
    ║        DAY TRADING ORCHESTRATOR SYSTEM v1.0                  ║
    ║        Multi-Broker • AI-Powered • Risk Management           ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_help():
    """Print help information"""
    help_text = """
    📖 Usage: python start_system.py [options]
    
    Options:
      --validate-only    Run only system validation (don't start application)
      --skip-validation  Skip system validation checks
      --help, -h         Show this help message
    
    🏁 Quick Start:
      python start_system.py              # Full validation + startup
      python start_system.py --validate-only    # Check system only
      python start_system.py --skip-validation  # Start without checks
    
    📝 Configuration:
      • Copy .env.example to .env and configure your API keys
      • Ensure all broker APIs are properly configured
      • Check logs/ directory for detailed system logs
    
    🔧 Troubleshooting:
      • Run validation script: python validate_system.py
      • Check system requirements in requirements.txt
      • Verify broker API credentials and permissions
    """
    print(help_text)


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Day Trading Orchestrator System Launcher",
        add_help=False
    )
    
    parser.add_argument('--validate-only', action='store_true',
                       help='Run only system validation')
    parser.add_argument('--skip-validation', action='store_true',
                       help='Skip system validation checks')
    parser.add_argument('--help', '-h', action='store_true',
                       help='Show help message')
    
    args = parser.parse_args()
    
    if args.help:
        print_banner()
        print_help()
        return True
    
    print_banner()
    
    # Print configuration info
    print(f"⚙️  Configuration:")
    print(f"   Environment: {settings.environment.value}")
    print(f"   Debug Mode: {settings.debug}")
    print(f"   Log Level: {settings.log_level}")
    print(f"   Database: {settings.db_type.value}")
    
    if settings.binance_api_key:
        print(f"   Binance: Configured")
    else:
        print(f"   Binance: Not configured")
    
    if settings.alpaca_api_key:
        print(f"   Alpaca: Configured")
    else:
        print(f"   Alpaca: Not configured")
    
    print(f"   AI Provider: {settings.ai_provider}")
    print(f"   Risk Management: Active")
    
    # Launch system
    launcher = SystemLauncher()
    success = await launcher.run(
        validate_only=args.validate_only,
        skip_validation=args.skip_validation
    )
    
    if success:
        print("\n👋 System shutdown complete. Goodbye!")
    else:
        print("\n💥 System startup failed!")
    
    return success


if __name__ == "__main__":
    try:
        # Set Windows event loop policy if needed
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        sys.exit(1)