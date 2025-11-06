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
    
    async def run_comprehensive_health_check(self) -> Dict[str, Any]:
        """Run all health checks and return comprehensive report"""
        print("🏥 Running Day Trading Orchestrator Health Check")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Platform: {platform.system()} {platform.release()}")
        print(f"Python: {platform.python_version()}")
        print("=" * 60)
        
        health_checks = [
            ("System Resources", self.check_system_resources),
            ("Network Connectivity", self.check_network_connectivity)
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

async def main():
    """Main health check runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Day Trading Orchestrator Health Check")
    parser.add_argument("--config", default="config.json", help="Configuration file path")
    parser.add_argument("--quick", action="store_true", help="Run quick health check only")
    
    args = parser.parse_args()
    
    # Create health monitor
    monitor = SystemHealthMonitor(args.config)
    
    # Run health check
    summary = await monitor.run_comprehensive_health_check()
    
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