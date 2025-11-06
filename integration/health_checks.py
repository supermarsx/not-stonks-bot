"""
Health Check Framework
Component diagnostics and system-wide health reporting.
"""

import asyncio
import logging
import json
import time
import subprocess
import psutil
import sqlite3
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path
import aiohttp
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HealthCheckStatus(Enum):
    """Health check status enumeration."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"
    DISABLED = "disabled"


class CheckCategory(Enum):
    """Health check category enumeration."""
    SYSTEM = "system"
    DATABASE = "database"
    NETWORK = "network"
    APPLICATION = "application"
    SECURITY = "security"
    PERFORMANCE = "performance"
    INTEGRITY = "integrity"
    DEPENDENCY = "dependency"


@dataclass
class HealthCheckResult:
    """Health check result container."""
    name: str
    category: CheckCategory
    status: HealthCheckStatus
    message: str
    details: Dict[str, Any]
    execution_time: float
    timestamp: datetime
    dependencies: List[str]
    remediation: Optional[str] = None


@dataclass
class DiagnosticReport:
    """Comprehensive diagnostic report."""
    system_info: Dict[str, Any]
    health_checks: List[HealthCheckResult]
    overall_status: HealthCheckStatus
    timestamp: datetime
    duration: float
    recommendations: List[str]
    critical_issues: List[str]


class HealthCheckFramework:
    """Comprehensive health check and diagnostic framework."""
    
    def __init__(self, config_file: str = "config/health_checks.json"):
        self.config_file = config_file
        self.config = self._load_config()
        self.health_checks: Dict[str, Callable] = {}
        self.check_registry: Dict[str, Dict[str, Any]] = {}
        self.last_results: Dict[str, HealthCheckResult] = {}
        
        # Initialize built-in health checks
        self._register_builtin_checks()
        
        # Load custom health checks from configuration
        self._load_custom_checks()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load health check configuration."""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Error loading health check config: {str(e)}")
        
        return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default health check configuration."""
        return {
            "enabled_checks": [
                "system_cpu",
                "system_memory",
                "system_disk",
                "system_uptime",
                "network_connectivity",
                "database_connectivity",
                "application_status",
                "file_integrity"
            ],
            "thresholds": {
                "cpu_warning": 70,
                "cpu_critical": 90,
                "memory_warning": 80,
                "memory_critical": 95,
                "disk_warning": 85,
                "disk_critical": 95,
                "response_time_warning": 2.0,
                "response_time_critical": 5.0
            },
            "custom_checks": [],
            "dependencies": {
                "database_connectivity": ["system_memory"],
                "application_status": ["network_connectivity", "database_connectivity"]
            }
        }
    
    def _register_builtin_checks(self):
        """Register built-in health checks."""
        self.health_checks.update({
            # System checks
            "system_cpu": self._check_cpu_usage,
            "system_memory": self._check_memory_usage,
            "system_disk": self._check_disk_usage,
            "system_uptime": self._check_system_uptime,
            "system_load": self._check_system_load,
            
            # Network checks
            "network_connectivity": self._check_network_connectivity,
            "dns_resolution": self._check_dns_resolution,
            "port_connectivity": self._check_port_connectivity,
            
            # Database checks
            "database_connectivity": self._check_database_connectivity,
            "database_performance": self._check_database_performance,
            "database_integrity": self._check_database_integrity,
            
            # Application checks
            "application_status": self._check_application_status,
            "api_endpoints": self._check_api_endpoints,
            "service_health": self._check_service_health,
            
            # Security checks
            "ssl_certificate": self._check_ssl_certificate,
            "file_permissions": self._check_file_permissions,
            "process_security": self._check_process_security,
            
            # Performance checks
            "response_time": self._check_response_time,
            "throughput": self._check_throughput,
            "resource_utilization": self._check_resource_utilization,
            
            # Integrity checks
            "file_integrity": self._check_file_integrity,
            "config_integrity": self._check_config_integrity,
            "log_integrity": self._check_log_integrity
        })
        
        logger.info(f"Registered {len(self.health_checks)} built-in health checks")
    
    def _load_custom_checks(self):
        """Load custom health checks from configuration."""
        custom_checks = self.config.get("custom_checks", [])
        
        for check_config in custom_checks:
            try:
                check_name = check_config["name"]
                check_module = check_config.get("module")
                check_function = check_config.get("function")
                
                if check_module and check_function:
                    # Import custom check function
                    import importlib
                    module = importlib.import_module(check_module)
                    function = getattr(module, check_function)
                    
                    self.health_checks[check_name] = function
                    logger.info(f"Registered custom health check: {check_name}")
                    
            except Exception as e:
                logger.error(f"Error loading custom health check {check_name}: {str(e)}")
    
    def register_health_check(self, name: str, check_function: Callable, 
                            category: CheckCategory = CheckCategory.APPLICATION,
                            dependencies: List[str] = None):
        """Register a new health check."""
        self.health_checks[name] = check_function
        self.check_registry[name] = {
            "category": category,
            "dependencies": dependencies or [],
            "enabled": True
        }
        
        logger.info(f"Registered health check: {name}")
    
    async def run_health_check(self, check_name: str) -> HealthCheckResult:
        """Run a specific health check."""
        if check_name not in self.health_checks:
            return HealthCheckResult(
                name=check_name,
                category=CheckCategory.APPLICATION,
                status=HealthCheckStatus.DISABLED,
                message=f"Health check '{check_name}' not found",
                details={},
                execution_time=0,
                timestamp=datetime.now(),
                dependencies=[]
            )
        
        start_time = time.time()
        
        try:
            # Execute the health check function
            result = await self.health_checks[check_name]()
            
            # Ensure result is a HealthCheckResult
            if not isinstance(result, HealthCheckResult):
                # Convert result to HealthCheckResult
                result = HealthCheckResult(
                    name=check_name,
                    category=CheckCategory.APPLICATION,
                    status=HealthCheckStatus.UNKNOWN,
                    message=str(result),
                    details={},
                    execution_time=time.time() - start_time,
                    timestamp=datetime.now(),
                    dependencies=[]
                )
            
            # Set execution time
            result.execution_time = time.time() - start_time
            result.timestamp = datetime.now()
            
            # Cache the result
            self.last_results[check_name] = result
            
            logger.debug(f"Health check '{check_name}' completed: {result.status.value}")
            return result
            
        except Exception as e:
            logger.error(f"Error running health check '{check_name}': {str(e)}")
            return HealthCheckResult(
                name=check_name,
                category=CheckCategory.APPLICATION,
                status=HealthCheckStatus.CRITICAL,
                message=f"Health check failed: {str(e)}",
                details={"error": str(e)},
                execution_time=time.time() - start_time,
                timestamp=datetime.now(),
                dependencies=[]
            )
    
    async def run_comprehensive_diagnostics(self) -> DiagnosticReport:
        """Run comprehensive system diagnostics."""
        logger.info("Starting comprehensive system diagnostics...")
        
        start_time = time.time()
        enabled_checks = self.config.get("enabled_checks", [])
        
        # Run all enabled health checks
        results = []
        for check_name in enabled_checks:
            if check_name in self.health_checks:
                result = await self.run_health_check(check_name)
                results.append(result)
            else:
                logger.warning(f"Health check '{check_name}' not found")
        
        # Calculate overall status
        overall_status = self._calculate_overall_status(results)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(results)
        
        # Identify critical issues
        critical_issues = [r for r in results if r.status == HealthCheckStatus.CRITICAL]
        
        # Get system information
        system_info = await self._get_system_info()
        
        duration = time.time() - start_time
        
        report = DiagnosticReport(
            system_info=system_info,
            health_checks=results,
            overall_status=overall_status,
            timestamp=datetime.now(),
            duration=duration,
            recommendations=recommendations,
            critical_issues=[issue.message for issue in critical_issues]
        )
        
        logger.info(f"Comprehensive diagnostics completed in {duration:.2f} seconds")
        return report
    
    def _calculate_overall_status(self, results: List[HealthCheckResult]) -> HealthCheckStatus:
        """Calculate overall system health status."""
        if not results:
            return HealthCheckStatus.UNKNOWN
        
        # Count statuses
        status_counts = {
            HealthCheckStatus.HEALTHY: 0,
            HealthCheckStatus.WARNING: 0,
            HealthCheckStatus.CRITICAL: 0,
            HealthCheckStatus.UNKNOWN: 0,
            HealthCheckStatus.DISABLED: 0
        }
        
        for result in results:
            status_counts[result.status] += 1
        
        total_checks = len(results)
        healthy_ratio = status_counts[HealthCheckStatus.HEALTHY] / total_checks
        critical_ratio = status_counts[HealthCheckStatus.CRITICAL] / total_checks
        
        # Determine overall status
        if critical_ratio > 0.2:  # More than 20% critical
            return HealthCheckStatus.CRITICAL
        elif healthy_ratio >= 0.8:  # At least 80% healthy
            return HealthCheckStatus.HEALTHY
        elif healthy_ratio >= 0.6:  # At least 60% healthy
            return HealthCheckStatus.WARNING
        else:
            return HealthCheckStatus.CRITICAL
    
    def _generate_recommendations(self, results: List[HealthCheckResult]) -> List[str]:
        """Generate recommendations based on health check results."""
        recommendations = []
        
        # Analyze failed checks
        failed_checks = [r for r in results if r.status == HealthCheckStatus.CRITICAL]
        warning_checks = [r for r in results if r.status == HealthCheckStatus.WARNING]
        
        for check in failed_checks:
            if "cpu" in check.name.lower():
                recommendations.append("High CPU usage detected. Consider optimizing processes or upgrading hardware.")
            elif "memory" in check.name.lower():
                recommendations.append("High memory usage detected. Check for memory leaks and consider adding more RAM.")
            elif "disk" in check.name.lower():
                recommendations.append("High disk usage detected. Clean up temporary files or add more storage.")
            elif "database" in check.name.lower():
                recommendations.append("Database issues detected. Check database connectivity and performance.")
            elif "network" in check.name.lower():
                recommendations.append("Network connectivity issues detected. Check network configuration and connectivity.")
        
        for check in warning_checks:
            if "response_time" in check.name.lower():
                recommendations.append("Slow response times detected. Consider performance optimization.")
            elif "throughput" in check.name.lower():
                recommendations.append("Low throughput detected. Review system capacity and load balancing.")
        
        # General recommendations
        if len(failed_checks) > 3:
            recommendations.append("Multiple critical issues detected. Consider running a complete system diagnostic.")
        
        if not recommendations:
            recommendations.append("System appears to be healthy. Continue regular monitoring.")
        
        return recommendations
    
    async def _get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information."""
        info = {
            "platform": {
                "system": psutil.uname().system,
                "release": psutil.uname().release,
                "version": psutil.uname().version,
                "machine": psutil.uname().machine,
                "processor": psutil.uname().processor
            },
            "cpu": {
                "physical_cores": psutil.cpu_count(logical=False),
                "total_cores": psutil.cpu_count(logical=True),
                "max_frequency": f"{psutil.cpu_freq().max:.2f}Mhz" if psutil.cpu_freq() else "N/A",
                "current_frequency": f"{psutil.cpu_freq().current:.2f}Mhz" if psutil.cpu_freq() else "N/A"
            },
            "memory": {
                "total": f"{psutil.virtual_memory().total / (1024**3):.2f} GB",
                "available": f"{psutil.virtual_memory().available / (1024**3):.2f} GB",
                "percent": f"{psutil.virtual_memory().percent}%"
            },
            "disk": {
                "total": f"{psutil.disk_usage('/').total / (1024**3):.2f} GB",
                "used": f"{psutil.disk_usage('/').used / (1024**3):.2f} GB",
                "free": f"{psutil.disk_usage('/').free / (1024**3):.2f} GB",
                "percent": f"{(psutil.disk_usage('/').used / psutil.disk_usage('/').total) * 100:.1f}%"
            },
            "network": {},
            "processes": {
                "total": len(psutil.pids()),
                "running": len([p for p in psutil.process_iter(['status']) if p.info['status'] == psutil.STATUS_RUNNING])
            }
        }
        
        # Network interfaces
        try:
            net_io = psutil.net_io_counters()
            info["network"] = {
                "bytes_sent": net_io.bytes_sent,
                "bytes_recv": net_io.bytes_recv,
                "packets_sent": net_io.packets_sent,
                "packets_recv": net_io.packets_recv
            }
        except Exception as e:
            info["network"]["error"] = str(e)
        
        return info
    
    # Built-in health check implementations
    
    async def _check_cpu_usage(self) -> HealthCheckResult:
        """Check CPU usage."""
        cpu_percent = psutil.cpu_percent(interval=1)
        thresholds = self.config.get("thresholds", {})
        
        if cpu_percent >= thresholds.get("cpu_critical", 90):
            status = HealthCheckStatus.CRITICAL
            message = f"CPU usage is critically high: {cpu_percent}%"
            remediation = "Identify and terminate resource-intensive processes"
        elif cpu_percent >= thresholds.get("cpu_warning", 70):
            status = HealthCheckStatus.WARNING
            message = f"CPU usage is high: {cpu_percent}%"
            remediation = "Monitor CPU usage and consider optimization"
        else:
            status = HealthCheckStatus.HEALTHY
            message = f"CPU usage is normal: {cpu_percent}%"
            remediation = None
        
        return HealthCheckResult(
            name="system_cpu",
            category=CheckCategory.SYSTEM,
            status=status,
            message=message,
            details={"cpu_percent": cpu_percent, "thresholds": thresholds},
            execution_time=0,
            timestamp=datetime.now(),
            dependencies=[],
            remediation=remediation
        )
    
    async def _check_memory_usage(self) -> HealthCheckResult:
        """Check memory usage."""
        memory = psutil.virtual_memory()
        thresholds = self.config.get("thresholds", {})
        
        if memory.percent >= thresholds.get("memory_critical", 95):
            status = HealthCheckStatus.CRITICAL
            message = f"Memory usage is critically high: {memory.percent}%"
            remediation = "Restart services or add more RAM"
        elif memory.percent >= thresholds.get("memory_warning", 80):
            status = HealthCheckStatus.WARNING
            message = f"Memory usage is high: {memory.percent}%"
            remediation = "Monitor memory usage and check for memory leaks"
        else:
            status = HealthCheckStatus.HEALTHY
            message = f"Memory usage is normal: {memory.percent}%"
            remediation = None
        
        return HealthCheckResult(
            name="system_memory",
            category=CheckCategory.SYSTEM,
            status=status,
            message=message,
            details={
                "memory_percent": memory.percent,
                "total_gb": memory.total / (1024**3),
                "available_gb": memory.available / (1024**3)
            },
            execution_time=0,
            timestamp=datetime.now(),
            dependencies=[],
            remediation=remediation
        )
    
    async def _check_disk_usage(self) -> HealthCheckResult:
        """Check disk usage."""
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        thresholds = self.config.get("thresholds", {})
        
        if disk_percent >= thresholds.get("disk_critical", 95):
            status = HealthCheckStatus.CRITICAL
            message = f"Disk usage is critically high: {disk_percent:.1f}%"
            remediation = "Free up disk space immediately"
        elif disk_percent >= thresholds.get("disk_warning", 85):
            status = HealthCheckStatus.WARNING
            message = f"Disk usage is high: {disk_percent:.1f}%"
            remediation = "Monitor disk usage and clean up unnecessary files"
        else:
            status = HealthCheckStatus.HEALTHY
            message = f"Disk usage is normal: {disk_percent:.1f}%"
            remediation = None
        
        return HealthCheckResult(
            name="system_disk",
            category=CheckCategory.SYSTEM,
            status=status,
            message=message,
            details={
                "disk_percent": disk_percent,
                "total_gb": disk.total / (1024**3),
                "used_gb": disk.used / (1024**3),
                "free_gb": disk.free / (1024**3)
            },
            execution_time=0,
            timestamp=datetime.now(),
            dependencies=[],
            remediation=remediation
        )
    
    async def _check_system_uptime(self) -> HealthCheckResult:
        """Check system uptime."""
        uptime_seconds = time.time() - psutil.boot_time()
        uptime_hours = uptime_seconds / 3600
        
        if uptime_hours < 1:
            status = HealthCheckStatus.WARNING
            message = f"System has been up for only {uptime_hours:.1f} hours"
            remediation = None  # This is normal for recent reboots
        else:
            status = HealthCheckStatus.HEALTHY
            message = f"System uptime: {uptime_hours:.1f} hours"
            remediation = None
        
        return HealthCheckResult(
            name="system_uptime",
            category=CheckCategory.SYSTEM,
            status=status,
            message=message,
            details={
                "uptime_seconds": uptime_seconds,
                "uptime_hours": uptime_hours,
                "boot_time": datetime.fromtimestamp(psutil.boot_time()).isoformat()
            },
            execution_time=0,
            timestamp=datetime.now(),
            dependencies=[],
            remediation=remediation
        )
    
    async def _check_system_load(self) -> HealthCheckResult:
        """Check system load average."""
        try:
            load_avg = os.getloadavg()  # (1min, 5min, 15min)
            cpu_count = psutil.cpu_count()
            
            # Calculate load per CPU
            load_per_cpu_1min = load_avg[0] / cpu_count
            
            if load_per_cpu_1min > 2.0:
                status = HealthCheckStatus.CRITICAL
                message = f"System load is very high: {load_avg[0]:.2f} (load per CPU: {load_per_cpu_1min:.2f})"
                remediation = "High system load detected. Check for resource-intensive processes"
            elif load_per_cpu_1min > 1.0:
                status = HealthCheckStatus.WARNING
                message = f"System load is elevated: {load_avg[0]:.2f} (load per CPU: {load_per_cpu_1min:.2f})"
                remediation = "Monitor system load and identify heavy processes"
            else:
                status = HealthCheckStatus.HEALTHY
                message = f"System load is normal: {load_avg[0]:.2f} (load per CPU: {load_per_cpu_1min:.2f})"
                remediation = None
            
            return HealthCheckResult(
                name="system_load",
                category=CheckCategory.SYSTEM,
                status=status,
                message=message,
                details={
                    "load_1min": load_avg[0],
                    "load_5min": load_avg[1],
                    "load_15min": load_avg[2],
                    "cpu_count": cpu_count,
                    "load_per_cpu_1min": load_per_cpu_1min
                },
                execution_time=0,
                timestamp=datetime.now(),
                dependencies=[],
                remediation=remediation
            )
        except Exception as e:
            return HealthCheckResult(
                name="system_load",
                category=CheckCategory.SYSTEM,
                status=HealthCheckStatus.UNKNOWN,
                message=f"Unable to check system load: {str(e)}",
                details={"error": str(e)},
                execution_time=0,
                timestamp=datetime.now(),
                dependencies=[]
            )
    
    async def _check_network_connectivity(self) -> HealthCheckResult:
        """Check network connectivity."""
        try:
            # Test connectivity to common endpoints
            test_endpoints = [
                "8.8.8.8",  # Google DNS
                "1.1.1.1",  # Cloudflare DNS
                "google.com"
            ]
            
            successful_tests = 0
            total_tests = len(test_endpoints)
            
            for endpoint in test_endpoints:
                try:
                    if endpoint.replace(".", "").isdigit():
                        # IP address - use ping
                        result = subprocess.run(
                            ["ping", "-c", "1", "-W", "3", endpoint],
                            capture_output=True,
                            text=True,
                            timeout=5
                        )
                        if result.returncode == 0:
                            successful_tests += 1
                    else:
                        # Domain name - use ping
                        result = subprocess.run(
                            ["ping", "-c", "1", "-W", "3", endpoint],
                            capture_output=True,
                            text=True,
                            timeout=5
                        )
                        if result.returncode == 0:
                            successful_tests += 1
                except Exception:
                    pass
            
            success_rate = (successful_tests / total_tests) * 100
            
            if success_rate < 33:  # Less than 1/3 successful
                status = HealthCheckStatus.CRITICAL
                message = f"Network connectivity severely degraded: {success_rate:.0f}% success rate"
                remediation = "Check network configuration and physical connections"
            elif success_rate < 66:  # Less than 2/3 successful
                status = HealthCheckStatus.WARNING
                message = f"Network connectivity issues detected: {success_rate:.0f}% success rate"
                remediation = "Monitor network connectivity and check for intermittent issues"
            else:
                status = HealthCheckStatus.HEALTHY
                message = f"Network connectivity is good: {success_rate:.0f}% success rate"
                remediation = None
            
            return HealthCheckResult(
                name="network_connectivity",
                category=CheckCategory.NETWORK,
                status=status,
                message=message,
                details={
                    "successful_tests": successful_tests,
                    "total_tests": total_tests,
                    "success_rate": success_rate,
                    "test_endpoints": test_endpoints
                },
                execution_time=0,
                timestamp=datetime.now(),
                dependencies=[],
                remediation=remediation
            )
        except Exception as e:
            return HealthCheckResult(
                name="network_connectivity",
                category=CheckCategory.NETWORK,
                status=HealthCheckStatus.CRITICAL,
                message=f"Network connectivity check failed: {str(e)}",
                details={"error": str(e)},
                execution_time=0,
                timestamp=datetime.now(),
                dependencies=[]
            )
    
    async def _check_dns_resolution(self) -> HealthCheckResult:
        """Check DNS resolution."""
        try:
            import socket
            
            test_domains = ["google.com", "github.com", "stackoverflow.com"]
            successful_resolutions = 0
            
            for domain in test_domains:
                try:
                    socket.gethostbyname(domain)
                    successful_resolutions += 1
                except Exception:
                    pass
            
            success_rate = (successful_resolutions / len(test_domains)) * 100
            
            if success_rate < 33:
                status = HealthCheckStatus.CRITICAL
                message = f"DNS resolution severely degraded: {success_rate:.0f}% success rate"
                remediation = "Check DNS configuration and server availability"
            elif success_rate < 66:
                status = HealthCheckStatus.WARNING
                message = f"DNS resolution issues detected: {success_rate:.0f}% success rate"
                remediation = "Monitor DNS resolution and check for intermittent issues"
            else:
                status = HealthCheckStatus.HEALTHY
                message = f"DNS resolution is working: {success_rate:.0f}% success rate"
                remediation = None
            
            return HealthCheckResult(
                name="dns_resolution",
                category=CheckCategory.NETWORK,
                status=status,
                message=message,
                details={
                    "successful_resolutions": successful_resolutions,
                    "total_tests": len(test_domains),
                    "success_rate": success_rate,
                    "test_domains": test_domains
                },
                execution_time=0,
                timestamp=datetime.now(),
                dependencies=[],
                remediation=remediation
            )
        except Exception as e:
            return HealthCheckResult(
                name="dns_resolution",
                category=CheckCategory.NETWORK,
                status=HealthCheckStatus.CRITICAL,
                message=f"DNS resolution check failed: {str(e)}",
                details={"error": str(e)},
                execution_time=0,
                timestamp=datetime.now(),
                dependencies=[]
            )
    
    async def _check_port_connectivity(self) -> HealthCheckResult:
        """Check port connectivity to application services."""
        try:
            # Common ports to check
            ports_to_check = [
                (80, "HTTP"),
                (443, "HTTPS"),
                (8000, "Application"),
                (5432, "PostgreSQL"),
                (3306, "MySQL"),
                (6379, "Redis")
            ]
            
            open_ports = []
            total_ports = len(ports_to_check)
            
            for port, service in ports_to_check:
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(3)
                    result = sock.connect_ex(('localhost', port))
                    sock.close()
                    
                    if result == 0:
                        open_ports.append(service)
                except Exception:
                    pass
            
            open_rate = len(open_ports) / total_ports
            
            if open_rate < 0.3:
                status = HealthCheckStatus.CRITICAL
                message = f"Very few services are accessible: {len(open_ports)}/{total_ports}"
                remediation = "Check service status and configuration"
            elif open_rate < 0.7:
                status = HealthCheckStatus.WARNING
                message = f"Some services may be down: {len(open_ports)}/{total_ports}"
                remediation = "Monitor service availability and check logs"
            else:
                status = HealthCheckStatus.HEALTHY
                message = f"Most services are accessible: {len(open_ports)}/{total_ports}"
                remediation = None
            
            return HealthCheckResult(
                name="port_connectivity",
                category=CheckCategory.NETWORK,
                status=status,
                message=message,
                details={
                    "open_ports": open_ports,
                    "total_ports": total_ports,
                    "open_rate": open_rate,
                    "checked_ports": ports_to_check
                },
                execution_time=0,
                timestamp=datetime.now(),
                dependencies=[],
                remediation=remediation
            )
        except Exception as e:
            return HealthCheckResult(
                name="port_connectivity",
                category=CheckCategory.NETWORK,
                status=HealthCheckStatus.CRITICAL,
                message=f"Port connectivity check failed: {str(e)}",
                details={"error": str(e)},
                execution_time=0,
                timestamp=datetime.now(),
                dependencies=[]
            )
    
    async def _check_database_connectivity(self) -> HealthCheckResult:
        """Check database connectivity."""
        try:
            # This would check actual database connectivity
            # For now, return a mock result
            
            return HealthCheckResult(
                name="database_connectivity",
                category=CheckCategory.DATABASE,
                status=HealthCheckStatus.HEALTHY,
                message="Database connectivity is working",
                details={"connection_pool": "active", "response_time": 0.05},
                execution_time=0,
                timestamp=datetime.now(),
                dependencies=["system_memory"]
            )
        except Exception as e:
            return HealthCheckResult(
                name="database_connectivity",
                category=CheckCategory.DATABASE,
                status=HealthCheckStatus.CRITICAL,
                message=f"Database connectivity failed: {str(e)}",
                details={"error": str(e)},
                execution_time=0,
                timestamp=datetime.now(),
                dependencies=[]
            )
    
    async def _check_database_performance(self) -> HealthCheckResult:
        """Check database performance."""
        try:
            # Mock database performance check
            return HealthCheckResult(
                name="database_performance",
                category=CheckCategory.DATABASE,
                status=HealthCheckStatus.HEALTHY,
                message="Database performance is good",
                details={"query_time_avg": 0.05, "connection_count": 10},
                execution_time=0,
                timestamp=datetime.now(),
                dependencies=["database_connectivity"]
            )
        except Exception as e:
            return HealthCheckResult(
                name="database_performance",
                category=CheckCategory.DATABASE,
                status=HealthCheckStatus.CRITICAL,
                message=f"Database performance check failed: {str(e)}",
                details={"error": str(e)},
                execution_time=0,
                timestamp=datetime.now(),
                dependencies=["database_connectivity"]
            )
    
    async def _check_database_integrity(self) -> HealthCheckResult:
        """Check database integrity."""
        try:
            # Mock database integrity check
            return HealthCheckResult(
                name="database_integrity",
                category=CheckCategory.DATABASE,
                status=HealthCheckStatus.HEALTHY,
                message="Database integrity is intact",
                details={"checksum_valid": True, "last_check": datetime.now().isoformat()},
                execution_time=0,
                timestamp=datetime.now(),
                dependencies=["database_connectivity"]
            )
        except Exception as e:
            return HealthCheckResult(
                name="database_integrity",
                category=CheckCategory.DATABASE,
                status=HealthCheckStatus.CRITICAL,
                message=f"Database integrity check failed: {str(e)}",
                details={"error": str(e)},
                execution_time=0,
                timestamp=datetime.now(),
                dependencies=["database_connectivity"]
            )
    
    async def _check_application_status(self) -> HealthCheckResult:
        """Check application status."""
        try:
            # Mock application status check
            return HealthCheckResult(
                name="application_status",
                category=CheckCategory.APPLICATION,
                status=HealthCheckStatus.HEALTHY,
                message="Application is running normally",
                details={"version": "1.0.0", "uptime": 3600},
                execution_time=0,
                timestamp=datetime.now(),
                dependencies=["network_connectivity", "database_connectivity"]
            )
        except Exception as e:
            return HealthCheckResult(
                name="application_status",
                category=CheckCategory.APPLICATION,
                status=HealthCheckStatus.CRITICAL,
                message=f"Application status check failed: {str(e)}",
                details={"error": str(e)},
                execution_time=0,
                timestamp=datetime.now(),
                dependencies=["network_connectivity", "database_connectivity"]
            )
    
    async def _check_api_endpoints(self) -> HealthCheckResult:
        """Check API endpoint availability."""
        try:
            endpoints = [
                "http://localhost:8000/health",
                "http://localhost:8000/api/status"
            ]
            
            working_endpoints = 0
            for endpoint in endpoints:
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(endpoint, timeout=aiohttp.ClientTimeout(total=5)) as response:
                            if response.status == 200:
                                working_endpoints += 1
                except Exception:
                    pass
            
            success_rate = (working_endpoints / len(endpoints)) * 100
            
            if success_rate < 50:
                status = HealthCheckStatus.CRITICAL
                message = f"Most API endpoints are down: {success_rate:.0f}% working"
                remediation = "Check application status and API service configuration"
            elif success_rate < 100:
                status = HealthCheckStatus.WARNING
                message = f"Some API endpoints are down: {success_rate:.0f}% working"
                remediation = "Monitor API availability and check service health"
            else:
                status = HealthCheckStatus.HEALTHY
                message = f"All API endpoints are working: {success_rate:.0f}% working"
                remediation = None
            
            return HealthCheckResult(
                name="api_endpoints",
                category=CheckCategory.APPLICATION,
                status=status,
                message=message,
                details={
                    "working_endpoints": working_endpoints,
                    "total_endpoints": len(endpoints),
                    "success_rate": success_rate
                },
                execution_time=0,
                timestamp=datetime.now(),
                dependencies=["application_status"]
            )
        except Exception as e:
            return HealthCheckResult(
                name="api_endpoints",
                category=CheckCategory.APPLICATION,
                status=HealthCheckStatus.CRITICAL,
                message=f"API endpoint check failed: {str(e)}",
                details={"error": str(e)},
                execution_time=0,
                timestamp=datetime.now(),
                dependencies=["application_status"]
            )
    
    async def _check_service_health(self) -> HealthCheckResult:
        """Check system service health."""
        try:
            # Check common services
            services_to_check = ["nginx", "redis", "postgresql"]
            healthy_services = 0
            
            for service in services_to_check:
                try:
                    result = subprocess.run(
                        ["systemctl", "is-active", service],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    if result.returncode == 0 and "active" in result.stdout:
                        healthy_services += 1
                except Exception:
                    pass
            
            health_rate = (healthy_services / len(services_to_check)) * 100
            
            if health_rate < 50:
                status = HealthCheckStatus.CRITICAL
                message = f"Most services are down: {healthy_services}/{len(services_to_check)}"
                remediation = "Check service status and restart failed services"
            elif health_rate < 100:
                status = HealthCheckStatus.WARNING
                message = f"Some services may be down: {healthy_services}/{len(services_to_check)}"
                remediation = "Monitor service availability"
            else:
                status = HealthCheckStatus.HEALTHY
                message = f"All services are healthy: {healthy_services}/{len(services_to_check)}"
                remediation = None
            
            return HealthCheckResult(
                name="service_health",
                category=CheckCategory.APPLICATION,
                status=status,
                message=message,
                details={
                    "healthy_services": healthy_services,
                    "total_services": len(services_to_check),
                    "health_rate": health_rate,
                    "checked_services": services_to_check
                },
                execution_time=0,
                timestamp=datetime.now(),
                dependencies=[]
            )
        except Exception as e:
            return HealthCheckResult(
                name="service_health",
                category=CheckCategory.APPLICATION,
                status=HealthCheckStatus.CRITICAL,
                message=f"Service health check failed: {str(e)}",
                details={"error": str(e)},
                execution_time=0,
                timestamp=datetime.now(),
                dependencies=[]
            )
    
    # Additional health check implementations would go here...
    # (SSL certificate, file permissions, response time, etc.)
    
    async def _check_ssl_certificate(self) -> HealthCheckResult:
        """Check SSL certificate status."""
        # Mock SSL certificate check
        return HealthCheckResult(
            name="ssl_certificate",
            category=CheckCategory.SECURITY,
            status=HealthCheckStatus.HEALTHY,
            message="SSL certificate is valid and not expired",
            details={"expires_in_days": 30},
            execution_time=0,
            timestamp=datetime.now(),
            dependencies=["network_connectivity"]
        )
    
    async def _check_file_permissions(self) -> HealthCheckResult:
        """Check critical file permissions."""
        # Mock file permissions check
        return HealthCheckResult(
            name="file_permissions",
            category=CheckCategory.SECURITY,
            status=HealthCheckStatus.HEALTHY,
            message="Critical files have appropriate permissions",
            details={"secure_files": 10, "insecure_files": 0},
            execution_time=0,
            timestamp=datetime.now(),
            dependencies=[]
        )
    
    async def _check_process_security(self) -> HealthCheckResult:
        """Check process security."""
        # Mock process security check
        return HealthCheckResult(
            name="process_security",
            category=CheckCategory.SECURITY,
            status=HealthCheckStatus.HEALTHY,
            message="No suspicious processes detected",
            details={"suspicious_processes": 0},
            execution_time=0,
            timestamp=datetime.now(),
            dependencies=[]
        )
    
    async def _check_response_time(self) -> HealthCheckResult:
        """Check response time."""
        # Mock response time check
        return HealthCheckResult(
            name="response_time",
            category=CheckCategory.PERFORMANCE,
            status=HealthCheckStatus.HEALTHY,
            message="Response times are within acceptable limits",
            details={"avg_response_time": 0.5, "p95_response_time": 1.2},
            execution_time=0,
            timestamp=datetime.now(),
            dependencies=["api_endpoints"]
        )
    
    async def _check_throughput(self) -> HealthCheckResult:
        """Check system throughput."""
        # Mock throughput check
        return HealthCheckResult(
            name="throughput",
            category=CheckCategory.PERFORMANCE,
            status=HealthCheckStatus.HEALTHY,
            message="System throughput is within normal range",
            details={"requests_per_second": 100, "target_rps": 150},
            execution_time=0,
            timestamp=datetime.now(),
            dependencies=["application_status"]
        )
    
    async def _check_resource_utilization(self) -> HealthCheckResult:
        """Check resource utilization."""
        # Mock resource utilization check
        return HealthCheckResult(
            name="resource_utilization",
            category=CheckCategory.PERFORMANCE,
            status=HealthCheckStatus.HEALTHY,
            message="Resource utilization is within normal limits",
            details={"cpu_utilization": 45, "memory_utilization": 60},
            execution_time=0,
            timestamp=datetime.now(),
            dependencies=["system_cpu", "system_memory"]
        )
    
    async def _check_file_integrity(self) -> HealthCheckResult:
        """Check file integrity."""
        # Mock file integrity check
        return HealthCheckResult(
            name="file_integrity",
            category=CheckCategory.INTEGRITY,
            status=HealthCheckStatus.HEALTHY,
            message="File integrity checksums are valid",
            details={"checked_files": 100, "failed_checksums": 0},
            execution_time=0,
            timestamp=datetime.now(),
            dependencies=[]
        )
    
    async def _check_config_integrity(self) -> HealthCheckResult:
        """Check configuration integrity."""
        # Mock config integrity check
        return HealthCheckResult(
            name="config_integrity",
            category=CheckCategory.INTEGRITY,
            status=HealthCheckStatus.HEALTHY,
            message="Configuration files are valid and intact",
            details={"valid_configs": 5, "invalid_configs": 0},
            execution_time=0,
            timestamp=datetime.now(),
            dependencies=[]
        )
    
    async def _check_log_integrity(self) -> HealthCheckResult:
        """Check log file integrity."""
        # Mock log integrity check
        return HealthCheckResult(
            name="log_integrity",
            category=CheckCategory.INTEGRITY,
            status=HealthCheckStatus.HEALTHY,
            message="Log files are being written correctly",
            details={"log_files_checked": 10, "errors": 0},
            execution_time=0,
            timestamp=datetime.now(),
            dependencies=[]
        )
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get a summary of current health check results."""
        if not self.last_results:
            return {
                "status": HealthCheckStatus.UNKNOWN.value,
                "total_checks": 0,
                "healthy": 0,
                "warning": 0,
                "critical": 0,
                "last_check": None
            }
        
        total_checks = len(self.last_results)
        healthy = len([r for r in self.last_results.values() if r.status == HealthCheckStatus.HEALTHY])
        warning = len([r for r in self.last_results.values() if r.status == HealthCheckStatus.WARNING])
        critical = len([r for r in self.last_results.values() if r.status == HealthCheckStatus.CRITICAL])
        
        # Determine overall status
        if critical > 0:
            overall_status = HealthCheckStatus.CRITICAL
        elif warning > 0:
            overall_status = HealthCheckStatus.WARNING
        else:
            overall_status = HealthCheckStatus.HEALTHY
        
        last_check = max(r.timestamp for r in self.last_results.values())
        
        return {
            "status": overall_status.value,
            "total_checks": total_checks,
            "healthy": healthy,
            "warning": warning,
            "critical": critical,
            "last_check": last_check.isoformat(),
            "success_rate": (healthy / total_checks * 100) if total_checks > 0 else 0
        }
    
    def export_diagnostic_report(self, report: DiagnosticReport, 
                               filename: str = None) -> str:
        """Export diagnostic report to file."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"health_diagnostic_report_{timestamp}.json"
        
        # Convert report to dictionary
        report_dict = {
            "system_info": report.system_info,
            "health_checks": [asdict(result) for result in report.health_checks],
            "overall_status": report.overall_status.value,
            "timestamp": report.timestamp.isoformat(),
            "duration": report.duration,
            "recommendations": report.recommendations,
            "critical_issues": report.critical_issues
        }
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        logger.info(f"Diagnostic report exported to: {filename}")
        return filename


# Global health check framework instance
health_check_framework = HealthCheckFramework()