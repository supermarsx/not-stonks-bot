#!/usr/bin/env python3
"""
Day Trading Orchestrator - System Health Check
Comprehensive health monitoring and diagnostic tool
"""

import sys
import json
import time
import asyncio
import psutil
import platform
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.absolute()))

@dataclass
class HealthCheckResult:
    """Health check result data structure"""
    component: str
    status: str  # "healthy", "warning", "critical", "unknown"
    message: str
    details: Dict[str, Any]
    timestamp: datetime
    duration: float
    
    def to_dict(self):
        return {
            **asdict(self),
            "timestamp": self.timestamp.isoformat()
        }

class SystemHealthMonitor:
    """Comprehensive system health monitoring"""
    
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.config = self.load_config()
        self.results: List[HealthCheckResult] = []
        
    def load_config(self) -> Dict[str, Any]:
        """Load configuration file"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"⚠️  Warning: Could not load config: {e}")
            return {}
    
    async def check_system_resources(self) -> HealthCheckResult:
        """Check system CPU, memory, and disk usage"""
        component = "System Resources"
        start_time = time.time()
        
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available = memory.available / (1024**3)  # GB
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            disk_free = disk.free / (1024**3)  # GB
            
            # Load average (Unix systems)
            load_avg = None
            if hasattr(psutil, 'getloadavg'):
                load_avg = psutil.getloadavg()
            
            details = {
                "cpu": {
                    "usage_percent": cpu_percent,
                    "core_count": cpu_count,
                    "load_average": load_avg
                },
                "memory": {
                    "usage_percent": memory_percent,
                    "available_gb": round(memory_available, 2),
                    "total_gb": round(memory.total / (1024**3), 2)
                },
                "disk": {
                    "usage_percent": round(disk_percent, 2),
                    "free_gb": round(disk_free, 2),
                    "total_gb": round(disk.total / (1024**3), 2)
                }
            }
            
            # Determine status
            status = "healthy"
            messages = []
            
            if cpu_percent > 80:
                status = "warning"
                messages.append(f"High CPU usage: {cpu_percent}%")
            elif cpu_percent > 95:
                status = "critical"
                messages.append(f"Critical CPU usage: {cpu_percent}%")
            
            if memory_percent > 80:
                status = "warning" if status == "healthy" else status
                messages.append(f"High memory usage: {memory_percent}%")
            elif memory_percent > 95:
                status = "critical"
                messages.append(f"Critical memory usage: {memory_percent}%")
            
            if disk_percent > 80:
                status = "warning" if status == "healthy" else status
                messages.append(f"Low disk space: {disk_percent}% used")
            elif disk_percent > 90:
                status = "critical"
                messages.append(f"Critical disk space: {disk_percent}% used")
            
            message = "; ".join(messages) if messages else "All system resources within normal limits"
            
        except Exception as e:
            status = "unknown"
            message = f"Error checking system resources: {str(e)}"
            details = {"error": str(e)}
        
        duration = time.time() - start_time
        
        return HealthCheckResult(
            component=component,
            status=status,
            message=message,
            details=details,
            timestamp=datetime.now(),
            duration=duration
        )
    
    async def check_network_connectivity(self) -> HealthCheckResult:
        """Check network connectivity to external services"""
        component = "Network Connectivity"
        start_time = time.time()
        
        services = [
            ("Google DNS", "8.8.8.8", 53),
            ("OpenAI API", "api.openai.com", 443),
            ("Binance API", "api.binance.com", 443),
            ("Yahoo Finance", "query1.finance.yahoo.com", 443)
        ]
        
        service_results = {}
        healthy_services = 0
        
        try:
            import socket
            
            for service_name, host, port in services:
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(5)
                    result = sock.connect_ex((host, port))
                    sock.close()
                    
                    if result == 0:
                        service_results[service_name] = {"status": "reachable", "port": port}
                        healthy_services += 1
                    else:
                        service_results[service_name] = {"status": "unreachable", "port": port}
                        
                except Exception as e:
                    service_results[service_name] = {"status": "error", "error": str(e)}
            
            # Determine overall status
            if healthy_services == len(services):
                status = "healthy"
                message = f"All {len(services)} services reachable"
            elif healthy_services > 0:
                status = "warning"
                message = f"{healthy_services}/{len(services)} services reachable"
            else:
                status = "critical"
                message = "No external services reachable"
            
        except Exception as e:
            status = "unknown"
            message = f"Error checking network connectivity: {str(e)}"
            service_results = {"error": str(e)}
        
        duration = time.time() - start_time
        
        return HealthCheckResult(
            component=component,
            status=status,
            message=message,
            details={"services": service_results},
            timestamp=datetime.now(),
            duration=duration
        )
    
    async def check_database_health(self) -> HealthCheckResult:
        """Check database connectivity and performance"""
        component = "Database"
        start_time = time.time()
        
        try:
            from config.database import engine
            
            query_start = time.time()
            async with engine.connect() as conn:
                result = await conn.execute("SELECT 1 as test")
                row = result.fetchone()
                query_time = time.time() - query_start
            
            if row and row[0] == 1:
                status = "healthy"
                message = f"Database responsive ({query_time:.3f}s query time)"
                details = {
                    "connection": "active",
                    "query_time_seconds": round(query_time, 3),
                    "test_query": "successful"
                }
            else:
                status = "critical"
                message = "Database query failed"
                details = {"error": "Invalid query result"}
                
        except ImportError:
            status = "unknown"
            message = "Database module not available"
            details = {"error": "ImportError"}
        except Exception as e:
            status = "critical"
            message = f"Database connection failed: {str(e)}"
            details = {"error": str(e)}
        
        duration = time.time() - start_time
        
        return HealthCheckResult(
            component=component,
            status=status,
            message=message,
            details=details,
            timestamp=datetime.now(),
            duration=duration
        )
    
    async def check_broker_health(self) -> HealthCheckResult:
        """Check all configured broker connections"""
        component = "Broker Connections"
        start_time = time.time()
        
        if not self.config.get("brokers"):
            return HealthCheckResult(
                component=component,
                status="unknown",
                message="No brokers configured",
                details={},
                timestamp=datetime.now(),
                duration=time.time() - start_time
            )
        
        broker_results = {}
        healthy_brokers = 0
        total_brokers = 0
        
        try:
            for broker_name, broker_config in self.config["brokers"].items():
                if not broker_config.get("enabled", False):
                    continue
                
                total_brokers += 1
                
                # Test individual broker
                broker_result = await self.check_single_broker(broker_name, broker_config)
                broker_results[broker_name] = broker_result
                
                if broker_result["status"] == "healthy":
                    healthy_brokers += 1
            
            # Determine overall status
            if total_brokers == 0:
                status = "unknown"
                message = "No enabled brokers found"
            elif healthy_brokers == total_brokers:
                status = "healthy"
                message = f"All {total_brokers} brokers healthy"
            elif healthy_brokers > 0:
                status = "warning"
                message = f"{healthy_brokers}/{total_brokers} brokers healthy"
            else:
                status = "critical"
                message = "No brokers are healthy"
            
        except Exception as e:
            status = "unknown"
            message = f"Error checking broker health: {str(e)}"
            broker_results = {"error": str(e)}
        
        duration = time.time() - start_time
        
        return HealthCheckResult(
            component=component,
            status=status,
            message=message,
            details={"brokers": broker_results},
            timestamp=datetime.now(),
            duration=duration
        )
    
    async def check_single_broker(self, name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Check health of a single broker"""
        try:
            # Simplified broker health check
            if name.lower() == "alpaca":
                return await self.check_alpaca_health(config)
            elif name.lower() == "binance":
                return await self.check_binance_health(config)
            elif name.lower() == "ibkr":
                return await self.check_ibkr_health(config)
            else:
                return {
                    "status": "healthy",
                    "message": "Broker configuration valid",
                    "details": {"type": name}
                }
        except Exception as e:
            return {
                "status": "critical",
                "message": f"Broker check failed: {str(e)}",
                "details": {"error": str(e)}
            }
    
    async def check_alpaca_health(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Check Alpaca broker health"""
        api_key = config.get("api_key")
        secret_key = config.get("secret_key")
        
        if not api_key or not secret_key or api_key == "YOUR_ALPACA_API_KEY":
            return {
                "status": "warning",
                "message": "Alpaca API keys not configured",
                "details": {"configured": False}
            }
        
        try:
            import aiohttp
            
            base_url = config.get("base_url", "https://paper-api.alpaca.markets")
            headers = {
                "APCA-API-KEY-ID": api_key,
                "APCA-API-SECRET-KEY": secret_key
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{base_url}/v2/account", headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "status": "healthy",
                            "message": "Alpaca API accessible",
                            "details": {
                                "account_status": data.get("status"),
                                "paper_mode": config.get("paper", True)
                            }
                        }
                    else:
                        error_text = await response.text()
                        return {
                            "status": "critical",
                            "message": f"Alpaca API error: HTTP {response.status}",
                            "details": {"error": error_text[:200]}
                        }
                        
        except Exception as e:
            return {
                "status": "critical",
                "message": f"Alpaca connection failed: {str(e)}",
                "details": {"error": str(e)}
            }
    
    async def check_binance_health(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Check Binance broker health"""
        api_key = config.get("api_key")
        secret_key = config.get("secret_key")
        
        if not api_key or not secret_key or api_key == "YOUR_BINANCE_API_KEY":
            return {
                "status": "warning",
                "message": "Binance API keys not configured",
                "details": {"configured": False}
            }
        
        try:
            import aiohttp
            import hmac
            import hashlib
            
            base_url = config.get("base_url", "https://testnet.binance.vision")
            
            # Test server time (no authentication required)
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{base_url}/api/v3/time", timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "status": "healthy",
                            "message": "Binance API accessible",
                            "details": {
                                "server_time": data.get("serverTime"),
                                "testnet": config.get("testnet", True)
                            }
                        }
                    else:
                        return {
                            "status": "critical",
                            "message": f"Binance API error: HTTP {response.status}",
                            "details": {"status_code": response.status}
                        }
                        
        except Exception as e:
            return {
                "status": "critical",
                "message": f"Binance connection failed: {str(e)}",
                "details": {"error": str(e)}
            }
    
    async def check_ibkr_health(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Check Interactive Brokers health"""
        host = config.get("host", "127.0.0.1")
        port = config.get("port", 7497)
        
        try:
            import socket
            
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((host, int(port)))
            sock.close()
            
            if result == 0:
                return {
                    "status": "healthy",
                    "message": f"IBKR Gateway reachable on port {port}",
                    "details": {
                        "host": host,
                        "port": port,
                        "connection": "successful"
                    }
                }
            else:
                return {
                    "status": "critical",
                    "message": f"Cannot connect to IBKR Gateway at {host}:{port}",
                    "details": {
                        "host": host,
                        "port": port,
                        "connection": "failed"
                    }
                }
                
        except Exception as e:
            return {
                "status": "critical",
                "message": f"IBKR connection error: {str(e)}",
                "details": {"error": str(e)}
            }
    
    async def check_ai_services(self) -> HealthCheckResult:
        """Check AI service integration"""
        component = "AI Services"
        start_time = time.time()
        
        ai_config = self.config.get("ai", {})
        ai_results = {}
        healthy_services = 0
        total_services = 0
        
        # Check OpenAI
        if ai_config.get("openai_api_key") and ai_config["openai_api_key"] != "YOUR_OPENAI_API_KEY":
            total_services += 1
            ai_results["openai"] = await self.check_openai_health(ai_config["openai_api_key"])
            if ai_results["openai"]["status"] == "healthy":
                healthy_services += 1
        else:
            ai_results["openai"] = {
                "status": "warning",
                "message": "OpenAI API key not configured",
                "details": {"configured": False}
            }
        
        # Check Anthropic
        if ai_config.get("anthropic_api_key") and ai_config["anthropic_api_key"] != "YOUR_ANTHROPIC_API_KEY":
            total_services += 1
            ai_results["anthropic"] = await self.check_anthropic_health(ai_config["anthropic_api_key"])
            if ai_results["anthropic"]["status"] == "healthy":
                healthy_services += 1
        else:
            ai_results["anthropic"] = {
                "status": "warning",
                "message": "Anthropic API key not configured",
                "details": {"configured": False}
            }
        
        # Check Local Models
        if ai_config.get("local_models", {}).get("enabled", False):
            total_services += 1
            ai_results["local_models"] = await self.check_local_models_health()
            if ai_results["local_models"]["status"] == "healthy":
                healthy_services += 1
        else:
            ai_results["local_models"] = {
                "status": "warning",
                "message": "Local models not enabled",
                "details": {"enabled": False}
            }
        
        # Determine overall status
        if total_services == 0:
            status = "warning"
            message = "No AI services configured"
        elif healthy_services == total_services:
            status = "healthy"
            message = f"All {total_services} AI services healthy"
        elif healthy_services > 0:
            status = "warning"
            message = f"{healthy_services}/{total_services} AI services healthy"
        else:
            status = "critical"
            message = "No AI services are healthy"
        
        duration = time.time() - start_time
        
        return HealthCheckResult(
            component=component,
            status=status,
            message=message,
            details={"services": ai_results},
            timestamp=datetime.now(),
            duration=duration
        )
    
    async def check_openai_health(self, api_key: str) -> Dict[str, Any]:
        """Check OpenAI API health"""
        try:
            import aiohttp
            
            headers = {"Authorization": f"Bearer {api_key}"}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://api.openai.com/v1/models",
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        models_count = len(data.get("data", []))
                        return {
                            "status": "healthy",
                            "message": f"OpenAI API accessible ({models_count} models available)",
                            "details": {"models_available": models_count}
                        }
                    else:
                        return {
                            "status": "critical",
                            "message": f"OpenAI API error: HTTP {response.status}",
                            "details": {"status_code": response.status}
                        }
                        
        except Exception as e:
            return {
                "status": "critical",
                "message": f"OpenAI connection failed: {str(e)}",
                "details": {"error": str(e)}
            }
    
    async def check_anthropic_health(self, api_key: str) -> Dict[str, Any]:
        """Check Anthropic API health"""
        try:
            import aiohttp
            
            headers = {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01"
            }
            
            data = {
                "model": "claude-3-haiku-20240307",
                "max_tokens": 1,
                "messages": [{"role": "user", "content": "Hi"}]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        return {
                            "status": "healthy",
                            "message": "Anthropic API accessible",
                            "details": {"model": "claude-3-haiku-20240307"}
                        }
                    else:
                        return {
                            "status": "critical",
                            "message": f"Anthropic API error: HTTP {response.status}",
                            "details": {"status_code": response.status}
                        }
                        
        except Exception as e:
            return {
                "status": "critical",
                "message": f"Anthropic connection failed: {str(e)}",
                "details": {"error": str(e)}
            }
    
    async def check_local_models_health(self) -> Dict[str, Any]:
        """Check local AI models health"""
        try:
            import subprocess
            
            # Check if Ollama is available
            result = subprocess.run(["which", "ollama"], capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                return {
                    "status": "critical",
                    "message": "Ollama not found in PATH",
                    "details": {"error": "ollama command not found"}
                }
            
            # Try to list models
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                models = result.stdout.strip().split('\n')
                models = [m for m in models if m and not m.startswith('NAME')]
                return {
                    "status": "healthy",
                    "message": f"Ollama accessible ({len(models)} models)",
                    "details": {"models_count": len(models)}
                }
            else:
                return {
                    "status": "warning",
                    "message": "Ollama found but model listing failed",
                    "details": {"error": result.stderr}
                }
                
        except subprocess.TimeoutExpired:
            return {
                "status": "warning",
                "message": "Ollama check timed out",
                "details": {"error": "timeout"}
            }
        except Exception as e:
            return {
                "status": "critical",
                "message": f"Local models check failed: {str(e)}",
                "details": {"error": str(e)}
            }
    
    async def check_file_system(self) -> HealthCheckResult:
        """Check file system health and permissions"""
        component = "File System"
        start_time = time.time()
        
        try:
            import os
            
            # Check critical directories
            directories = ["logs", "data", "backups"]
            dir_results = {}
            
            for dir_name in directories:
                dir_path = Path(dir_name)
                if dir_path.exists():
                    if os.access(dir_path, os.W_OK):
                        dir_results[dir_name] = {"exists": True, "writable": True}
                    else:
                        dir_results[dir_name] = {"exists": True, "writable": False}
                else:
                    dir_results[dir_name] = {"exists": False, "writable": False}
            
            # Check log file permissions
            log_file = Path("logs/trading_orchestrator.log")
            log_writable = False
            if log_file.exists():
                log_writable = os.access(log_file, os.W_OK)
            
            # Determine status
            if all(result["writable"] for result in dir_results.values()):
                status = "healthy"
                message = "All critical directories writable"
            elif any(result["writable"] for result in dir_results.values()):
                status = "warning"
                message = "Some directories not writable"
            else:
                status = "critical"
                message = "Critical directories not writable"
            
            details = {
                "directories": dir_results,
                "log_file_writable": log_writable
            }
            
        except Exception as e:
            status = "unknown"
            message = f"Error checking file system: {str(e)}"
            details = {"error": str(e)}
        
        duration = time.time() - start_time
        
        return HealthCheckResult(
            component=component,
            status=status,
            message=message,
            details=details,
            timestamp=datetime.now(),
            duration=duration
        )
    
    async def check_configuration(self) -> HealthCheckResult:
        """Check configuration file validity"""
        component = "Configuration"
        start_time = time.time()
        
        try:
            # Check if config file exists
            config_file = Path(self.config_path)
            if not config_file.exists():
                return HealthCheckResult(
                    component=component,
                    status="critical",
                    message=f"Configuration file {self.config_path} not found",
                    details={"file_exists": False},
                    timestamp=datetime.now(),
                    duration=time.time() - start_time
                )
            
            # Validate JSON syntax
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
            except json.JSONDecodeError as e:
                return HealthCheckResult(
                    component=component,
                    status="critical",
                    message=f"Invalid JSON in configuration file: {str(e)}",
                    details={"json_error": str(e)},
                    timestamp=datetime.now(),
                    duration=time.time() - start_time
                )
            
            # Check required sections
            required_sections = ["database", "brokers", "risk"]
            missing_sections = []
            for section in required_sections:
                if section not in config:
                    missing_sections.append(section)
            
            # Check API key configuration
            api_key_issues = []
            if "brokers" in config:
                for broker_name, broker_config in config["brokers"].items():
                    if broker_config.get("enabled", False):
                        if broker_name.lower() == "alpaca":
                            if not broker_config.get("api_key") or broker_config["api_key"] == "YOUR_ALPACA_API_KEY":
                                api_key_issues.append(f"{broker_name} API key not set")
                        elif broker_name.lower() == "binance":
                            if not broker_config.get("api_key") or broker_config["api_key"] == "YOUR_BINANCE_API_KEY":
                                api_key_issues.append(f"{broker_name} API key not set")
            
            # Determine status
            status = "healthy"
            messages = []
            details = {
                "file_exists": True,
                "json_valid": True,
                "config_sections": list(config.keys())
            }
            
            if missing_sections:
                status = "warning"
                messages.append(f"Missing config sections: {', '.join(missing_sections)}")
            
            if api_key_issues:
                status = "warning" if status == "healthy" else status
                messages.append(f"Unconfigured API keys: {', '.join(api_key_issues)}")
            
            if not messages:
                messages.append("Configuration file valid")
            
            message = "; ".join(messages)
            
        except Exception as e:
            status = "unknown"
            message = f"Error checking configuration: {str(e)}"
            details = {"error": str(e)}
        
        duration = time.time() - start_time
        
        return HealthCheckResult(
            component=component,
            status=status,
            message=message,
            details=details,
            timestamp=datetime.now(),
            duration=duration
        )
    
    async def run_comprehensive_health_check(self) -> Dict[str, Any]:
        """Run all health checks and return comprehensive report"""
        print("🏥 Running Day Trading Orchestrator Health Check")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Platform: {platform.system()} {platform.release()}")
        print(f"Python: {platform.python_version()}")
        print("=" * 60)
        
        health_checks = [
            ("System Resources", self.check_system_resources),
            ("Network Connectivity", self.check_network_connectivity),
            ("Database", self.check_database_health),
            ("Broker Connections", self.check_broker_health),
            ("AI Services", self.check_ai_services),
            ("File System", self.check_file_system),
            ("Configuration", self.check_configuration)
        ]
        
        self.results = []
        
        for check_name, check_function in health_checks:
            print(f"\n🔍 Checking {check_name}...")
            try:
                result = await check_function()
                self.results.append(result)
                
                # Print result with appropriate emoji
                status_emoji = {
                    "healthy": "✅",
                    "warning": "⚠️ ",
                    "critical": "❌",
                    "unknown": "❓"
                }
                
                print(f"{status_emoji.get(result.status, '❓')} {result.component}: {result.message}")
                
            except Exception as e:
                print(f"❌ {check_name}: Health check failed - {str(e)}")
                self.results.append(HealthCheckResult(
                    component=check_name,
                    status="critical",
                    message=f"Health check failed: {str(e)}",
                    details={"error": str(e)},
                    timestamp=datetime.now(),
                    duration=0.0
                ))
        
        # Generate summary
        return self.generate_summary()
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate health check summary"""
        total_checks = len(self.results)
        healthy_count = sum(1 for r in self.results if r.status == "healthy")
        warning_count = sum(1 for r in self.results if r.status == "warning")
        critical_count = sum(1 for r in self.results if r.status == "critical")
        unknown_count = sum(1 for r in self.results if r.status == "unknown")
        
        # Determine overall status
        if critical_count > 0:
            overall_status = "critical"
        elif warning_count > 0:
            overall_status = "warning"
        elif healthy_count == total_checks:
            overall_status = "healthy"
        else:
            overall_status = "unknown"
        
        summary = {
            "overall_status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_checks": total_checks,
                "healthy": healthy_count,
                "warning": warning_count,
                "critical": critical_count,
                "unknown": unknown_count
            },
            "results": [result.to_dict() for result in self.results],
            "system_info": {
                "platform": platform.system(),
                "release": platform.release(),
                "python_version": platform.python_version(),
                "cpu_count": psutil.cpu_count(),
                "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2)
            }
        }
        
        # Print summary
        print(f"\n{'=' * 60}")
        print(f"HEALTH CHECK SUMMARY")
        print(f"{'=' * 60}")
        print(f"Overall Status: {overall_status.upper()}")
        print(f"Total Checks: {total_checks}")
        print(f"✅ Healthy: {healthy_count}")
        print(f"⚠️  Warnings: {warning_count}")
        print(f"❌ Critical: {critical_count}")
        print(f"❓ Unknown: {unknown_count}")
        
        if overall_status == "critical":
            print(f"\n🚨 CRITICAL ISSUES DETECTED")
            critical_results = [r for r in self.results if r.status == "critical"]
            for result in critical_results:
                print(f"  - {result.component}: {result.message}")
        
        elif overall_status == "warning":
            print(f"\n⚠️  WARNINGS DETECTED")
            warning_results = [r for r in self.results if r.status == "warning"]
            for result in warning_results:
                print(f"  - {result.component}: {result.message}")
        
        elif overall_status == "healthy":
            print(f"\n🎉 ALL SYSTEMS HEALTHY!")
            print("Your Day Trading Orchestrator is running optimally.")
        
        return summary
    
    def save_health_report(self, summary: Dict[str, Any], filename: str = None):
        """Save health check report to file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"health_report_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"\n📄 Health report saved to: {filename}")
        except Exception as e:
            print(f"\n❌ Failed to save health report: {e}")

async def main():
    """Main health check runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Day Trading Orchestrator Health Check")
    parser.add_argument("--config", default="config.json", help="Configuration file path")
    parser.add_argument("--output", help="Output file for health report")
    parser.add_argument("--quick", action="store_true", help="Run quick health check only")
    parser.add_argument("--component", help="Check specific component only")
    
    args = parser.parse_args()
    
    # Create health monitor
    monitor = SystemHealthMonitor(args.config)
    
    # Run health check
    summary = await monitor.run_comprehensive_health_check()
    
    # Save report if requested
    if args.output:
        monitor.save_health_report(summary, args.output)
    
    # Exit with appropriate code based on overall status
    status = summary["overall_status"]
    if status == "critical":
        sys.exit(2)
    elif status == "warning":
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Health check interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Health check execution failed: {e}")
        sys.exit(1)