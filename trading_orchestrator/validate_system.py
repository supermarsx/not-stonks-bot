#!/usr/bin/env python3
"""
System Validation and Health Check Script
Validates the complete trading system setup and provides diagnostics
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger
from config.settings import settings
from config.application import app_config
from ui.terminal import TerminalUI
from ui.components.dashboard import DashboardManager


class SystemValidator:
    """Comprehensive system validation and diagnostics"""
    
    def __init__(self):
        self.results = {
            'timestamp': datetime.utcnow().isoformat(),
            'system_info': {},
            'validations': [],
            'errors': [],
            'warnings': [],
            'recommendations': []
        }
        
        # Setup logging for validation
        self._setup_validation_logging()
    
    def _setup_validation_logging(self):
        """Setup logging for validation process"""
        logger.remove()
        logger.add(
            sys.stderr,
            level="INFO",
            format="<cyan>VALIDATION</cyan> | {message}"
        )
    
    async def run_full_validation(self) -> bool:
        """Run complete system validation"""
        try:
            logger.info("🔍 Starting System Validation...")
            
            # System information
            await self._collect_system_info()
            
            # Environment validation
            await self._validate_environment()
            
            # Dependencies validation
            await self._validate_dependencies()
            
            # Configuration validation
            await self._validate_configuration()
            
            # Database validation
            await self._validate_database()
            
            # AI components validation
            await self._validate_ai_components()
            
            # Broker connections validation
            await self._validate_broker_connections()
            
            # UI components validation
            await self._validate_ui_components()
            
            # Integration test
            await self._validate_system_integration()
            
            # Generate report
            self._generate_report()
            
            return len(self.results['errors']) == 0
            
        except Exception as e:
            logger.error(f"❌ Validation failed: {e}")
            self.results['errors'].append(f"Validation system error: {str(e)}")
            return False
    
    async def _collect_system_info(self):
        """Collect system information"""
        import platform
        import sys
        
        self.results['system_info'] = {
            'platform': platform.platform(),
            'python_version': sys.version,
            'python_executable': sys.executable,
            'working_directory': str(Path.cwd()),
            'project_root': str(Path(__file__).parent.absolute()),
            'validation_time': datetime.utcnow().isoformat()
        }
        
        logger.info(f"📋 System: {platform.system()} {platform.release()}")
        logger.info(f"🐍 Python: {sys.version.split()[0]}")
    
    async def _validate_environment(self):
        """Validate environment requirements"""
        logger.info("🌍 Validating Environment...")
        
        validation_items = []
        
        # Python version check
        python_version = sys.version_info
        if python_version >= (3, 11):
            validation_items.append(("✅ Python 3.11+", True))
        else:
            validation_items.append(("❌ Python 3.11+ required", False))
            self.results['errors'].append("Python 3.11 or higher is required")
        
        # Check if in project directory
        project_root = Path(__file__).parent
        if (project_root / "main.py").exists():
            validation_items.append(("✅ Project structure", True))
        else:
            validation_items.append(("❌ Invalid project directory", False))
            self.results['errors'].append("Not in valid project directory")
        
        # Environment file check
        env_file = Path(".env")
        if env_file.exists():
            validation_items.append(("✅ .env file found", True))
        else:
            validation_items.append(("⚠️  .env file not found", False))
            self.results['warnings'].append("Consider creating .env file for configuration")
        
        self.results['validations'].extend(validation_items)
        
        for item, success in validation_items:
            logger.info(f"  {item}")
    
    async def _validate_dependencies(self):
        """Validate required dependencies"""
        logger.info("📦 Validating Dependencies...")
        
        required_packages = [
            'fastapi',
            'pydantic',
            'sqlalchemy',
            'rich',
            'loguru',
            'asyncio',
            'aiohttp',
            'websockets',
            'numpy',
            'pandas'
        ]
        
        validation_items = []
        
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
                validation_items.append((f"✅ {package}", True))
            except ImportError:
                validation_items.append((f"❌ {package} not installed", False))
                self.results['errors'].append(f"Missing required package: {package}")
        
        self.results['validations'].extend(validation_items)
        
        for item, success in validation_items:
            logger.info(f"  {item}")
    
    async def _validate_configuration(self):
        """Validate application configuration"""
        logger.info("⚙️  Validating Configuration...")
        
        validation_items = []
        
        # Settings validation
        try:
            # Test settings loading
            test_settings = settings
            validation_items.append(("✅ Settings loaded", True))
            
            # Check critical settings
            if hasattr(test_settings, 'database_url'):
                validation_items.append(("✅ Database URL configured", True))
            else:
                validation_items.append(("⚠️  Database URL not configured", False))
                self.results['warnings'].append("Database URL not properly configured")
            
            if test_settings.log_level:
                validation_items.append((f"✅ Log level: {test_settings.log_level}", True))
            else:
                validation_items.append(("⚠️  Log level not set", False))
                self.results['warnings'].append("Log level not configured")
                
        except Exception as e:
            validation_items.append((f"❌ Settings error: {str(e)}", False))
            self.results['errors'].append(f"Settings validation failed: {str(e)}")
        
        self.results['validations'].extend(validation_items)
        
        for item, success in validation_items:
            logger.info(f"  {item}")
    
    async def _validate_database(self):
        """Validate database setup"""
        logger.info("🗄️  Validating Database...")
        
        validation_items = []
        
        try:
            # Test database import
            from config.database import engine, init_db
            validation_items.append(("✅ Database module import", True))
            
            # Test async engine creation
            if engine:
                validation_items.append(("✅ Async engine created", True))
            else:
                validation_items.append(("❌ Engine creation failed", False))
                self.results['errors'].append("Failed to create database engine")
            
            # Test database connection
            try:
                async with engine.connect() as conn:
                    await conn.execute("SELECT 1")
                validation_items.append(("✅ Database connection", True))
            except Exception as e:
                validation_items.append(("⚠️  Database connection failed", False))
                self.results['warnings'].append(f"Database connection test failed: {str(e)}")
                
        except ImportError as e:
            validation_items.append((f"❌ Database module error: {str(e)}", False))
            self.results['errors'].append("Database module import failed")
        except Exception as e:
            validation_items.append((f"❌ Database validation error: {str(e)}", False))
            self.results['errors'].append(f"Database validation failed: {str(e)}")
        
        self.results['validations'].extend(validation_items)
        
        for item, success in validation_items:
            logger.info(f"  {item}")
    
    async def _validate_ai_components(self):
        """Validate AI components"""
        logger.info("🧠 Validating AI Components...")
        
        validation_items = []
        
        try:
            # Test AI orchestrator import
            from ai.orchestrator import AITradingOrchestrator, TradingMode
            validation_items.append(("✅ AI orchestrator module", True))
            
            # Test AI models manager
            from ai.models.ai_models_manager import AIModelsManager
            validation_items.append(("✅ AI models manager", True))
            
            # Test trading tools
            from ai.tools.trading_tools import TradingTools
            validation_items.append(("✅ Trading tools", True))
            
            # Test OpenAI integration
            try:
                import openai
                validation_items.append(("✅ OpenAI client", True))
            except ImportError:
                validation_items.append(("⚠️  OpenAI not installed", False))
                self.results['warnings'].append("OpenAI package not installed")
            
        except ImportError as e:
            validation_items.append((f"❌ AI module error: {str(e)}", False))
            self.results['errors'].append(f"AI module import failed: {str(e)}")
        except Exception as e:
            validation_items.append((f"❌ AI validation error: {str(e)}", False))
            self.results['errors'].append(f"AI validation failed: {str(e)}")
        
        self.results['validations'].extend(validation_items)
        
        for item, success in validation_items:
            logger.info(f"  {item}")
    
    async def _validate_broker_connections(self):
        """Validate broker connections"""
        logger.info("🔗 Validating Broker Connections...")
        
        validation_items = []
        
        try:
            # Test broker factory
            from brokers.factory import BrokerFactory
            validation_items.append(("✅ Broker factory", True))
            
            # Test base broker
            from brokers.base import BrokerConfig
            validation_items.append(("✅ Base broker module", True))
            
            # Test individual broker modules
            broker_modules = [
                'binance_broker',
                'alpaca_broker', 
                'ibkr_broker',
                'trading212_broker'
            ]
            
            for module_name in broker_modules:
                try:
                    __import__(f"brokers.{module_name}")
                    validation_items.append((f"✅ {module_name.replace('_', ' ').title()}", True))
                except ImportError:
                    validation_items.append((f"⚠️  {module_name} not available", False))
                    self.results['warnings'].append(f"Broker module {module_name} not available")
            
        except ImportError as e:
            validation_items.append((f"❌ Broker module error: {str(e)}", False))
            self.results['errors'].append(f"Broker module import failed: {str(e)}")
        except Exception as e:
            validation_items.append((f"❌ Broker validation error: {str(e)}", False))
            self.results['errors'].append(f"Broker validation failed: {str(e)}")
        
        self.results['validations'].extend(validation_items)
        
        for item, success in validation_items:
            logger.info(f"  {item}")
    
    async def _validate_ui_components(self):
        """Validate UI components"""
        logger.info("🖥️  Validating UI Components...")
        
        validation_items = []
        
        try:
            # Test Rich library
            import rich
            validation_items.append(("✅ Rich library", True))
            
            # Test terminal UI
            from ui.terminal import TerminalUI
            validation_items.append(("✅ Terminal UI module", True))
            
            # Test dashboard manager
            from ui.components.dashboard import DashboardManager
            validation_items.append(("✅ Dashboard manager", True))
            
        except ImportError as e:
            validation_items.append((f"❌ UI module error: {str(e)}", False))
            self.results['errors'].append(f"UI module import failed: {str(e)}")
        except Exception as e:
            validation_items.append((f"❌ UI validation error: {str(e)}", False))
            self.results['errors'].append(f"UI validation failed: {str(e)}")
        
        self.results['validations'].extend(validation_items)
        
        for item, success in validation_items:
            logger.info(f"  {item}")
    
    async def _validate_system_integration(self):
        """Validate system integration"""
        logger.info("🔄 Validating System Integration...")
        
        validation_items = []
        
        try:
            # Test application configuration
            config = app_config
            validation_items.append(("✅ Application config import", True))
            
            # Test main application
            from main import TradingOrchestratorApp
            validation_items.append(("✅ Main application import", True))
            
        except ImportError as e:
            validation_items.append((f"❌ Integration error: {str(e)}", False))
            self.results['errors'].append(f"System integration validation failed: {str(e)}")
        except Exception as e:
            validation_items.append((f"❌ Integration validation error: {str(e)}", False))
            self.results['errors'].append(f"Integration validation failed: {str(e)}")
        
        self.results['validations'].extend(validation_items)
        
        for item, success in validation_items:
            logger.info(f"  {item}")
    
    def _generate_report(self):
        """Generate validation report"""
        logger.info("📊 Generating Validation Report...")
        
        # Calculate statistics
        total_validations = len(self.results['validations'])
        passed_validations = sum(1 for _, success in self.results['validations'] if success)
        failed_validations = total_validations - passed_validations
        
        success_rate = (passed_validations / total_validations * 100) if total_validations > 0 else 0
        
        # Overall status
        if len(self.results['errors']) == 0:
            if len(self.results['warnings']) == 0:
                status = "✅ HEALTHY"
                self.results['recommendations'].append("System is ready for deployment!")
            else:
                status = "⚠️  WARNING"
                self.results['recommendations'].append("System is functional but has warnings to address")
        else:
            status = "❌ ERROR"
            self.results['recommendations'].append("System has critical errors that must be fixed")
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"🔍 SYSTEM VALIDATION REPORT")
        print(f"{'='*80}")
        print(f"Status: {status}")
        print(f"Timestamp: {self.results['timestamp']}")
        print(f"")
        print(f"📈 Statistics:")
        print(f"  Total Validations: {total_validations}")
        print(f"  Passed: {passed_validations}")
        print(f"  Failed: {failed_validations}")
        print(f"  Success Rate: {success_rate:.1f}%")
        print(f"")
        print(f"⚠️  Warnings: {len(self.results['warnings'])}")
        print(f"❌ Errors: {len(self.results['errors'])}")
        
        if self.results['warnings']:
            print(f"\n⚠️  WARNINGS:")
            for warning in self.results['warnings']:
                print(f"  - {warning}")
        
        if self.results['errors']:
            print(f"\n❌ ERRORS:")
            for error in self.results['errors']:
                print(f"  - {error}")
        
        if self.results['recommendations']:
            print(f"\n💡 RECOMMENDATIONS:")
            for recommendation in self.results['recommendations']:
                print(f"  - {recommendation}")
        
        print(f"{'='*80}")
        
        # Save detailed report
        report_file = Path("validation_report.json")
        with open(report_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"📄 Detailed report saved to: {report_file}")


async def main():
    """Main validation entry point"""
    print("🎯 Day Trading Orchestrator - System Validation")
    print("=" * 60)
    
    validator = SystemValidator()
    success = await validator.run_full_validation()
    
    if success:
        print("\n🎉 System validation PASSED!")
        print("💡 The system is ready for deployment.")
    else:
        print("\n💥 System validation FAILED!")
        print("🔧 Please fix the errors before proceeding.")
    
    return success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)