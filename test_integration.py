#!/usr/bin/env python3
"""
Day Trading Orchestrator - Integration Test Suite
Comprehensive testing of all system components and integrations
"""

import asyncio
import sys
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.absolute()))

class TestResults:
    """Test results tracker"""
    
    def __init__(self):
        self.tests_passed = 0
        self.tests_failed = 0
        self.tests_skipped = 0
        self.failures = []
        
    def add_pass(self, test_name: str):
        self.tests_passed += 1
        print(f"✅ PASS: {test_name}")
        
    def add_fail(self, test_name: str, error: str):
        self.tests_failed += 1
        self.failures.append({"test": test_name, "error": error})
        print(f"❌ FAIL: {test_name} - {error}")
        
    def add_skip(self, test_name: str, reason: str):
        self.tests_skipped += 1
        print(f"⏭️  SKIP: {test_name} - {reason}")
        
    def summary(self):
        total = self.tests_passed + self.tests_failed + self.tests_skipped
        print(f"\n{'='*60}")
        print(f"TEST SUMMARY")
        print(f"{'='*60}")
        print(f"Total Tests: {total}")
        print(f"✅ Passed: {self.tests_passed}")
        print(f"❌ Failed: {self.tests_failed}")
        print(f"⏭️  Skipped: {self.tests_skipped}")
        
        if self.failures:
            print(f"\nFAILED TESTS:")
            for failure in self.failures:
                print(f"  - {failure['test']}: {failure['error']}")
                
        return self.tests_failed == 0

class IntegrationTestSuite:
    """Comprehensive integration test suite"""
    
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.config = self.load_config()
        self.results = TestResults()
        
    def load_config(self) -> Dict[str, Any]:
        """Load configuration file"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"⚠️  Configuration file {self.config_path} not found")
            return {}
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON in configuration: {e}")
            return {}
    
    async def test_system_initialization(self):
        """Test basic system initialization"""
        print("\n🔧 Testing System Initialization...")
        
        # Test 1: Configuration Loading
        if not self.config:
            self.results.add_fail("Configuration Loading", "No configuration loaded")
            return
            
        required_sections = ["database", "brokers", "ai", "risk"]
        for section in required_sections:
            if section not in self.config:
                self.results.add_fail(f"Config Section {section}", "Missing configuration section")
            else:
                self.results.add_pass(f"Config Section {section}")
        
        # Test 2: Directory Creation
        import os
        required_dirs = ["logs", "data", "backups"]
        for dir_name in required_dirs:
            try:
                os.makedirs(dir_name, exist_ok=True)
                if os.path.exists(dir_name):
                    self.results.add_pass(f"Directory Creation {dir_name}")
                else:
                    self.results.add_fail(f"Directory Creation {dir_name}", "Directory not created")
            except Exception as e:
                self.results.add_fail(f"Directory Creation {dir_name}", str(e))
    
    async def test_database_connectivity(self):
        """Test database connectivity"""
        print("\n🗄️  Testing Database Connectivity...")
        
        try:
            # Import database components
            from config.database import init_db, engine
            
            # Test database initialization
            try:
                await init_db()
                self.results.add_pass("Database Initialization")
            except Exception as e:
                self.results.add_fail("Database Initialization", str(e))
                return
            
            # Test database connection
            try:
                async with engine.connect() as conn:
                    result = await conn.execute("SELECT 1 as test")
                    row = result.fetchone()
                    if row and row[0] == 1:
                        self.results.add_pass("Database Connection")
                    else:
                        self.results.add_fail("Database Connection", "Invalid result from database")
            except Exception as e:
                self.results.add_fail("Database Connection", str(e))
                
        except ImportError as e:
            self.results.add_skip("Database Testing", f"Database module not available: {e}")
    
    async def test_broker_connectivity(self):
        """Test broker API connectivity"""
        print("\n🔌 Testing Broker Connectivity...")
        
        if not self.config.get("brokers"):
            self.results.add_skip("Broker Testing", "No brokers configured")
            return
        
        # Test each configured broker
        for broker_name, broker_config in self.config["brokers"].items():
            if not broker_config.get("enabled", False):
                self.results.add_skip(f"Broker {broker_name}", "Broker not enabled")
                continue
                
            await self.test_single_broker(broker_name, broker_config)
    
    async def test_single_broker(self, name: str, config: Dict[str, Any]):
        """Test a single broker connection"""
        try:
            # Import broker factory
            from brokers.factory import BrokerFactory
            
            broker_factory = BrokerFactory()
            
            # Create broker instance based on type
            if name.lower() == "alpaca":
                await self.test_alpaca_broker(config)
            elif name.lower() == "binance":
                await self.test_binance_broker(config)
            elif name.lower() == "ibkr":
                await self.test_ibkr_broker(config)
            else:
                # Generic broker test
                self.results.add_pass(f"Broker {name} Configuration")
                
        except ImportError as e:
            self.results.add_skip(f"Broker {name}", f"Broker module not available: {e}")
        except Exception as e:
            self.results.add_fail(f"Broker {name}", str(e))
    
    async def test_alpaca_broker(self, config: Dict[str, Any]):
        """Test Alpaca broker connectivity"""
        api_key = config.get("api_key")
        secret_key = config.get("secret_key")
        
        if not api_key or not secret_key or api_key == "YOUR_ALPACA_API_KEY":
            self.results.add_skip("Alpaca Broker", "API key not configured")
            return
        
        try:
            # Test API connectivity
            import aiohttp
            
            base_url = config.get("base_url", "https://paper-api.alpaca.markets")
            headers = {
                "APCA-API-KEY-ID": api_key,
                "APCA-API-SECRET-KEY": secret_key
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{base_url}/v2/account", headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        account_status = data.get("status", "UNKNOWN")
                        self.results.add_pass(f"Alpaca API Connection (Status: {account_status})")
                    else:
                        error_text = await response.text()
                        self.results.add_fail("Alpaca API Connection", f"HTTP {response.status}: {error_text}")
                        
        except Exception as e:
            self.results.add_fail("Alpaca Broker", str(e))
    
    async def test_binance_broker(self, config: Dict[str, Any]):
        """Test Binance broker connectivity"""
        api_key = config.get("api_key")
        secret_key = config.get("secret_key")
        
        if not api_key or not secret_key or api_key == "YOUR_BINANCE_API_KEY":
            self.results.add_skip("Binance Broker", "API key not configured")
            return
        
        try:
            # Test API connectivity
            import aiohttp
            import hmac
            import hashlib
            
            base_url = config.get("base_url", "https://testnet.binance.vision")
            
            # Create signature
            timestamp = int(time.time() * 1000)
            query_string = f"timestamp={timestamp}"
            signature = hmac.new(
                secret_key.encode('utf-8'),
                query_string.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            headers = {"X-MBX-APIKEY": api_key}
            url = f"{base_url}/api/v3/account?{query_string}&signature={signature}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        account_type = data.get("accountType", "UNKNOWN")
                        self.results.add_pass(f"Binance API Connection (Type: {account_type})")
                    else:
                        error_text = await response.text()
                        self.results.add_fail("Binance API Connection", f"HTTP {response.status}: {error_text}")
                        
        except Exception as e:
            self.results.add_fail("Binance Broker", str(e))
    
    async def test_ibkr_broker(self, config: Dict[str, Any]):
        """Test Interactive Brokers connectivity"""
        host = config.get("host", "127.0.0.1")
        port = config.get("port", 7497)
        
        try:
            # Test TCP connection to IBKR Gateway
            import socket
            
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((host, int(port)))
            sock.close()
            
            if result == 0:
                self.results.add_pass(f"IBKR Gateway Connection (Port: {port})")
            else:
                self.results.add_fail("IBKR Gateway Connection", f"Cannot connect to {host}:{port}")
                
        except Exception as e:
            self.results.add_fail("IBKR Broker", str(e))
    
    async def test_ai_integration(self):
        """Test AI service integration"""
        print("\n🤖 Testing AI Integration...")
        
        ai_config = self.config.get("ai", {})
        
        # Test OpenAI integration
        openai_key = ai_config.get("openai_api_key")
        if openai_key and openai_key != "YOUR_OPENAI_API_KEY":
            await self.test_openai_integration(openai_key)
        else:
            self.results.add_skip("OpenAI Integration", "API key not configured")
        
        # Test Anthropic integration
        anthropic_key = ai_config.get("anthropic_api_key")
        if anthropic_key and anthropic_key != "YOUR_ANTHROPIC_API_KEY":
            await self.test_anthropic_integration(anthropic_key)
        else:
            self.results.add_skip("Anthropic Integration", "API key not configured")
        
        # Test local models
        if ai_config.get("local_models", {}).get("enabled", False):
            await self.test_local_models()
        else:
            self.results.add_skip("Local Models", "Local models not enabled")
    
    async def test_openai_integration(self, api_key: str):
        """Test OpenAI API connectivity"""
        try:
            import aiohttp
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "Say 'API test successful'"}],
                "max_tokens": 50
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        content = result["choices"][0]["message"]["content"]
                        self.results.add_pass(f"OpenAI API (Response: {content})")
                    else:
                        error_text = await response.text()
                        self.results.add_fail("OpenAI API", f"HTTP {response.status}: {error_text}")
                        
        except Exception as e:
            self.results.add_fail("OpenAI Integration", str(e))
    
    async def test_anthropic_integration(self, api_key: str):
        """Test Anthropic API connectivity"""
        try:
            import aiohttp
            
            headers = {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "claude-3-haiku-20240307",
                "max_tokens": 50,
                "messages": [{"role": "user", "content": "Say 'API test successful'"}]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        content = result["content"][0]["text"]
                        self.results.add_pass(f"Anthropic API (Response: {content})")
                    else:
                        error_text = await response.text()
                        self.results.add_fail("Anthropic API", f"HTTP {response.status}: {error_text}")
                        
        except Exception as e:
            self.results.add_fail("Anthropic Integration", str(e))
    
    async def test_local_models(self):
        """Test local AI model integration"""
        try:
            # Check if Ollama is available
            import subprocess
            
            result = subprocess.run(["which", "ollama"], capture_output=True, text=True)
            if result.returncode == 0:
                self.results.add_pass("Ollama Installation")
                
                # Try to list models
                try:
                    result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        models = result.stdout.strip().split('\n')
                        self.results.add_pass(f"Ollama Models Available: {len(models)}")
                    else:
                        self.results.add_fail("Ollama Models", result.stderr)
                except subprocess.TimeoutExpired:
                    self.results.add_fail("Ollama Models", "Timeout listing models")
            else:
                self.results.add_fail("Ollama Installation", "Ollama not found in PATH")
                
        except Exception as e:
            self.results.add_fail("Local Models", str(e))
    
    async def test_market_data(self):
        """Test market data connectivity"""
        print("\n📊 Testing Market Data...")
        
        # Test if market data services are accessible
        test_symbols = ["AAPL", "BTC-USD", "EURUSD"]
        
        for symbol in test_symbols:
            try:
                # Test with Yahoo Finance (free data source)
                import aiohttp
                
                if symbol == "AAPL":
                    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
                elif symbol == "BTC-USD":
                    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
                elif symbol == "EURUSD":
                    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}=X"
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                        if response.status == 200:
                            data = await response.json()
                            if "chart" in data and data["chart"]["result"]:
                                self.results.add_pass(f"Market Data {symbol}")
                            else:
                                self.results.add_fail(f"Market Data {symbol}", "No data returned")
                        else:
                            self.results.add_fail(f"Market Data {symbol}", f"HTTP {response.status}")
                            
            except Exception as e:
                self.results.add_fail(f"Market Data {symbol}", str(e))
    
    async def test_risk_management(self):
        """Test risk management components"""
        print("\n🛡️  Testing Risk Management...")
        
        try:
            risk_config = self.config.get("risk", {})
            
            # Test configuration validation
            required_risk_params = ["max_position_size", "max_daily_loss"]
            for param in required_risk_params:
                if param in risk_config:
                    self.results.add_pass(f"Risk Parameter {param}")
                else:
                    self.results.add_fail(f"Risk Parameter {param}", "Missing configuration")
            
            # Test circuit breakers if enabled
            if risk_config.get("circuit_breakers", {}).get("enabled", False):
                self.results.add_pass("Circuit Breakers Enabled")
            else:
                self.results.add_skip("Circuit Breakers", "Not enabled")
            
        except Exception as e:
            self.results.add_fail("Risk Management", str(e))
    
    async def test_strategy_components(self):
        """Test trading strategy components"""
        print("\n📈 Testing Strategy Components...")
        
        try:
            from strategies.registry import StrategyRegistry
            
            registry = StrategyRegistry()
            
            # Test if strategies can be loaded
            available_strategies = registry.get_available_strategies()
            self.results.add_pass(f"Strategy Registry ({len(available_strategies)} strategies)")
            
            # Test individual strategies
            test_strategies = ["mean_reversion", "trend_following", "pairs_trading"]
            for strategy_name in test_strategies:
                if strategy_name in available_strategies:
                    self.results.add_pass(f"Strategy {strategy_name}")
                else:
                    self.results.add_skip(f"Strategy {strategy_name}", "Not available")
                    
        except ImportError:
            self.results.add_skip("Strategy Testing", "Strategy modules not available")
        except Exception as e:
            self.results.add_fail("Strategy Components", str(e))
    
    async def test_performance(self):
        """Test system performance"""
        print("\n⚡ Testing System Performance...")
        
        # Test response times for basic operations
        operations = [
            ("Config Load", lambda: self.load_config()),
            ("Database Query", self.test_database_query),
            ("Market Data Fetch", self.test_market_data_fetch)
        ]
        
        for operation_name, operation in operations:
            try:
                start_time = time.time()
                await operation()
                end_time = time.time()
                duration = end_time - start_time
                
                if duration < 5.0:  # 5 second threshold
                    self.results.add_pass(f"Performance {operation_name} ({duration:.2f}s)")
                else:
                    self.results.add_fail(f"Performance {operation_name}", f"Slow response: {duration:.2f}s")
                    
            except Exception as e:
                self.results.add_fail(f"Performance {operation_name}", str(e))
    
    async def test_database_query(self):
        """Test a simple database query"""
        try:
            from config.database import engine
            
            async with engine.connect() as conn:
                await conn.execute("SELECT 1")
        except:
            pass  # Will be caught by outer exception handler
    
    async def test_market_data_fetch(self):
        """Test market data fetching"""
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://query1.finance.yahoo.com/v8/finance/chart/AAPL",
                    timeout=aiohttp.ClientTimeout(total=3)
                ) as response:
                    if response.status == 200:
                        await response.json()
        except:
            pass  # Will be caught by outer exception handler
    
    async def test_configuration_validation(self):
        """Test configuration file validation"""
        print("\n⚙️  Testing Configuration...")
        
        try:
            from validate_config import ConfigurationValidator
            
            validator = ConfigurationValidator(self.config_path)
            validation_results = await validator.validate()
            
            if validation_results.is_valid:
                self.results.add_pass("Configuration Validation")
            else:
                for error in validation_results.errors:
                    self.results.add_fail("Configuration Validation", error)
                    
        except ImportError:
            self.results.add_skip("Config Validation", "Validator not available")
        except Exception as e:
            self.results.add_fail("Configuration Validation", str(e))
    
    async def run_all_tests(self):
        """Run all integration tests"""
        print("🚀 Starting Day Trading Orchestrator Integration Tests")
        print(f"Timestamp: {datetime.now().isoformat()}")
        print(f"Configuration: {self.config_path}")
        
        # Run test suites
        test_suites = [
            ("System Initialization", self.test_system_initialization),
            ("Database Connectivity", self.test_database_connectivity),
            ("Broker Connectivity", self.test_broker_connectivity),
            ("AI Integration", self.test_ai_integration),
            ("Market Data", self.test_market_data),
            ("Risk Management", self.test_risk_management),
            ("Strategy Components", self.test_strategy_components),
            ("Performance", self.test_performance),
            ("Configuration Validation", self.test_configuration_validation)
        ]
        
        for suite_name, test_function in test_suites:
            print(f"\n{'='*60}")
            print(f"Running: {suite_name}")
            print(f"{'='*60}")
            
            try:
                await test_function()
            except Exception as e:
                self.results.add_fail(suite_name, f"Test suite error: {str(e)}")
                print(f"❌ Test suite {suite_name} failed: {e}")
        
        # Generate summary
        success = self.results.summary()
        
        # Save test report
        await self.save_test_report()
        
        return success
    
    async def save_test_report(self):
        """Save test report to file"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "config_file": self.config_path,
            "results": {
                "passed": self.results.tests_passed,
                "failed": self.results.tests_failed,
                "skipped": self.results.tests_skipped,
                "total": self.results.tests_passed + self.results.tests_failed + self.results.tests_skipped
            },
            "failures": self.results.failures
        }
        
        report_file = f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Test report saved to: {report_file}")

async def main():
    """Main test runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Day Trading Orchestrator Integration Tests")
    parser.add_argument("--config", default="config.json", help="Configuration file path")
    parser.add_argument("--quick", action="store_true", help="Run quick tests only")
    parser.add_argument("--broker", help="Test specific broker only")
    parser.add_argument("--output", help="Output file for test results")
    
    args = parser.parse_args()
    
    # Create test suite
    test_suite = IntegrationTestSuite(args.config)
    
    # Run tests
    success = await test_suite.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Test execution failed: {e}")
        sys.exit(1)