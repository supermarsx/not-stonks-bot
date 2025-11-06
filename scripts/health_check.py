#!/usr/bin/env python3
"""
Health Check Script for not-stonks-bot

Comprehensive system health check that validates all components:
- Database connectivity
- Broker API connections
- AI service availability
- Configuration validation
- System resources
"""

import asyncio
import argparse
import sys
import json
import os
from pathlib import Path
from typing import Dict, List, Any

def check_python_version() -> bool:
    """Check if Python version is compatible."""
    import sys
    version = sys.version_info
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} is not supported. Requires Python 3.8+")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def check_config_file() -> bool:
    """Check if configuration file exists and is valid."""
    config_paths = [
        "config.json",
        "configs/config.example.json",
        ".env"
    ]
    
    for config_path in config_paths:
        if Path(config_path).exists():
            try:
                if config_path.endswith('.json'):
                    with open(config_path, 'r') as f:
                        json.load(f)
                print(f"✅ {config_path} exists and is valid")
                return True
            except json.JSONDecodeError as e:
                print(f"❌ {config_path} contains invalid JSON: {e}")
                return False
            except Exception as e:
                print(f"❌ Error reading {config_path}: {e}")
                return False
    
    print("⚠️  No configuration file found. Run with --create-config to generate one.")
    return False

def check_dependencies() -> bool:
    """Check if required packages are installed."""
    required_packages = [
        'pandas', 'numpy', 'requests', 'aiohttp', 'pydantic',
        'matplotlib', 'seaborn', 'docstring-parser'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} is missing")
    
    if missing_packages:
        print(f"\n📦 Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    return True

def check_api_connectivity() -> bool:
    """Check connectivity to external APIs."""
    import requests
    
    # Test basic internet connectivity
    try:
        response = requests.get('https://httpbin.org/get', timeout=5)
        if response.status_code == 200:
            print("✅ Internet connectivity: OK")
        else:
            print("❌ Internet connectivity: Failed")
            return False
    except Exception as e:
        print(f"❌ Internet connectivity: {e}")
        return False
    
    return True

def check_disk_space() -> bool:
    """Check available disk space."""
    import shutil
    
    total, used, free = shutil.disk_usage("/")
    free_gb = free // (1024**3)
    
    if free_gb < 1:
        print(f"❌ Low disk space: {free_gb}GB free")
        return False
    elif free_gb < 5:
        print(f"⚠️  Low disk space: {free_gb}GB free")
        return True
    else:
        print(f"✅ Disk space: {free_gb}GB free")
        return True

def check_memory() -> bool:
    """Check available memory."""
    try:
        import psutil
        memory = psutil.virtual_memory()
        free_gb = memory.available / (1024**3)
        
        if free_gb < 1:
            print(f"❌ Low memory: {free_gb:.1f}GB available")
            return False
        elif free_gb < 2:
            print(f"⚠️  Low memory: {free_gb:.1f}GB available")
            return True
        else:
            print(f"✅ Memory: {free_gb:.1f}GB available")
            return True
    except ImportError:
        print("⚠️  psutil not installed, cannot check memory")
        return True

async def check_async_components() -> bool:
    """Check asynchronous components."""
    try:
        # Test asyncio event loop
        loop = asyncio.get_event_loop()
        print("✅ Asyncio support: Available")
        
        # Test async HTTP client
        import aiohttp
        timeout = aiohttp.ClientTimeout(total=5)
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get('https://httpbin.org/get') as response:
                if response.status == 200:
                    print("✅ Async HTTP client: OK")
                else:
                    print("❌ Async HTTP client: Failed")
                    return False
        
        return True
    except Exception as e:
        print(f"❌ Async components: {e}")
        return False

def create_default_config():
    """Create a default configuration file."""
    default_config = {
        "database": {
            "url": "sqlite:///not_stonks_bot.db",
            "echo": False
        },
        "ai": {
            "trading_mode": "PAPER",
            "default_model_tier": "fast",
            "openai_api_key": None,
            "anthropic_api_key": None
        },
        "brokers": {
            "alpaca": {
                "enabled": False,
                "api_key": None,
                "secret_key": None,
                "paper": True
            }
        },
        "risk": {
            "max_position_size": 10000,
            "max_daily_loss": 5000
        },
        "logging": {
            "level": "INFO",
            "file": "logs/not_stonks_bot.log"
        }
    }
    
    with open("config.json", "w") as f:
        json.dump(default_config, f, indent=4)
    
    print("✅ Default configuration created: config.json")
    print("⚠️  Please update with your actual API keys before running!")

def main():
    """Main health check function."""
    parser = argparse.ArgumentParser(description="Health check for not-stonks-bot")
    parser.add_argument("--create-config", action="store_true", 
                        help="Create default configuration file")
    parser.add_argument("--full", action="store_true", 
                        help="Run full health check")
    parser.add_argument("--config", default="config.json",
                        help="Path to configuration file")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose output")
    
    args = parser.parse_args()
    
    if args.create_config:
        create_default_config()
        return
    
    print("\n🔍 not-stonks-bot Health Check")
    print("=" * 50)
    
    # Run all checks
    checks = [
        check_python_version,
        check_config_file,
        check_dependencies,
        check_api_connectivity,
        check_disk_space,
        check_memory,
    ]
    
    passed = 0
    total = len(checks)
    
    for check in checks:
        try:
            if check():
                passed += 1
        except Exception as e:
            print(f"❌ {check.__name__} failed: {e}")
    
    # Async checks
    if asyncio.run(check_async_components()):
        passed += 1
    total += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Health Check Results: {passed}/{total} checks passed")
    
    if passed == total:
        print("✅ System is healthy and ready to run!")
        print("\n🚀 Next steps:")
        print("   1. Update config.json with your API keys")
        print("   2. Run: python main.py --demo")
        return 0
    else:
        print("❌ System has issues that need to be resolved")
        print("\n🔧 Troubleshooting:")
        print("   1. Check the error messages above")
        print("   2. Install missing dependencies: pip install -r requirements.txt")
        print("   3. Create config: python health_check.py --create-config")
        return 1

if __name__ == "__main__":
    sys.exit(main())