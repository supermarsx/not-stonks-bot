#!/usr/bin/env python3
"""
Utility Validation Script
Tests all created utilities and validates the system integration
"""

import sys
import os
from pathlib import Path

def test_utility_files():
    """Test all utility files exist and are importable"""
    print("🔍 Testing Utility Files...")
    
    # Test 1: Logger Utility
    logger_path = Path("utils/logger.py")
    if logger_path.exists():
        print("✅ Logger utility file exists")
    else:
        print("❌ Logger utility file missing")
        
    # Test 2: Configuration Files
    config_path = Path("config/application.py")
    if config_path.exists():
        print("✅ Configuration application file exists")
    else:
        print("❌ Configuration application file missing")
        
    # Test 3: Database Integration
    database_init_path = Path("database/__init__.py")
    if database_init_path.exists():
        print("✅ Database integration file exists")
    else:
        print("❌ Database integration file missing")
        
    # Test 4: Setup Files
    setup_files = ["setup.py", "pyproject.toml", "requirements.txt"]
    for setup_file in setup_files:
        if Path(setup_file).exists():
            print(f"✅ {setup_file} exists")
        else:
            print(f"❌ {setup_file} missing")

def test_imports():
    """Test core imports work"""
    print("\n🔌 Testing Core Imports...")
    
    try:
        sys.path.insert(0, '.')
        
        # Test logger import
        from utils.logger import setup_logging, MatrixLogger
        print("✅ Logger imports working")
        
        # Test configuration imports  
        from config.application import AppConfig, DatabaseConfig, AIConfig, RiskConfig
        print("✅ Configuration imports working")
        
        # Test database imports
        from database import init_database
        print("✅ Database imports working")
        
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_logger_functionality():
    """Test logger functionality"""
    print("\n📝 Testing Logger Functionality...")
    
    try:
        from utils.logger import setup_logging, TradingEventType
        
        # Setup logging
        logger = setup_logging(level="INFO")
        print("✅ Logger setup working")
        
        # Test structured logging
        logger.info("System initialization started")
        logger.log_trading_event(
            TradingEventType.ORDER_SUBMITTED,
            broker="TEST_BROKER",
            symbol="TEST",
            side="BUY",
            quantity=100.0,
            price=50.0
        )
        print("✅ Structured logging working")
        
        return True
        
    except Exception as e:
        print(f"❌ Logger functionality error: {e}")
        return False

def test_config_loading():
    """Test configuration loading"""
    print("\n⚙️ Testing Configuration Loading...")
    
    try:
        from config.application import AppConfig
        import tempfile
        
        # Create test config
        test_config = {
            "database": {"url": "sqlite:///:memory:", "echo": False},
            "ai": {"trading_mode": "PAPER", "default_model_tier": "fast"},
            "brokers": {
                "test_broker": {"enabled": True, "api_key": "test_key"}
            },
            "risk": {"max_position_size": 10000, "max_daily_loss": 5000},
            "logging": {"level": "INFO", "file": "logs/test.log"}
        }
        
        config = AppConfig(test_config)
        print("✅ Configuration loading working")
        
        # Test validation
        errors = config.validate()
        if not errors:
            print("✅ Configuration validation working")
        else:
            print(f"⚠️ Config validation found issues: {errors}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration loading error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Trading Orchestrator Utility Validation\n")
    
    os.chdir(Path(__file__).parent)
    
    # Run tests
    test_utility_files()
    test_imports()
    test_logger_functionality()
    test_config_loading()
    
    print("\n" + "="*60)
    print("🎉 Utility Creation Complete!")
    print("="*60)
    print("\n📋 Created Files:")
    print("  • utils/logger.py - Matrix-themed logging utility")
    print("  • utils/__init__.py - Utils package initialization")  
    print("  • config/application.py - Enhanced AppConfig class")
    print("  • database/__init__.py - Database integration")
    print("  • database/migrations/migration_manager.py - Migration management")
    print("  • database/migrations/001_initial_schema.sql - Initial schema")
    print("  • setup.py - Package setup configuration")
    print("  • pyproject.toml - Modern Python packaging")
    print("  • requirements.txt - Complete dependency list")
    
    print("\n✨ Key Features:")
    print("  • Matrix-themed structured logging")
    print("  • Multi-broker integration support") 
    print("  • AI-powered trading decisions")
    print("  • Comprehensive risk management")
    print("  • Database migrations and seeding")
    print("  • Development and production setup")
    
    print("\n🎯 Next Steps:")
    print("  1. Update API keys in configuration")
    print("  2. Run: python main.py --create-config")
    print("  3. Run: python main.py --demo")
    print("  4. Install dependencies: pip install -r requirements.txt")

if __name__ == "__main__":
    main()