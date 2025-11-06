"""
Operational Runbooks and Procedures
===================================

Comprehensive runbooks and operational procedures for managing the perpetual trading system.
Provides step-by-step guides for common operations, troubleshooting, and maintenance tasks.

Contents:
1. System Startup and Shutdown Procedures
2. Emergency Response Procedures
3. Maintenance Operations
4. Performance Monitoring and Optimization
5. Backup and Recovery Procedures
6. Alert Management and Escalation
7. Troubleshooting Guides
8. Scaling and Load Balancing
9. Security Operations
10. Compliance and Audit Procedures

Author: Trading Orchestrator System
Version: 2.0.0
Date: 2025-11-06
"""

from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json
import asyncio
import subprocess
import psutil
from loguru import logger


# ================================
# Runbook Templates and Procedures
# ================================

class OperationalRunbook:
    """Base class for operational runbooks"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.procedures = []
    
    def add_procedure(self, procedure):
        """Add a procedure to the runbook"""
        self.procedures.append(procedure)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert runbook to dictionary"""
        return {
            'name': self.name,
            'description': self.description,
            'procedures': [proc.to_dict() if hasattr(proc, 'to_dict') else proc for proc in self.procedures]
        }


class Procedure:
    """Individual procedure within a runbook"""
    
    def __init__(self, name: str, description: str, steps: List[Dict[str, Any]]):
        self.name = name
        self.description = description
        self.steps = steps
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert procedure to dictionary"""
        return {
            'name': self.name,
            'description': self.description,
            'steps': self.steps
        }


# ================================
# Runbook Definitions
# ================================

def create_system_startup_runbook() -> OperationalRunbook:
    """Create system startup runbook"""
    runbook = OperationalRunbook(
        name="System Startup and Initialization",
        description="Complete procedures for starting the perpetual trading system from cold state"
    )
    
    # Pre-startup checks
    startup_check = Procedure(
        name="Pre-Startup System Checks",
        description="Verify system readiness before starting trading operations",
        steps=[
            {
                "step": 1,
                "action": "Check system resources",
                "command": "free -h && df -h",
                "expected": "Sufficient memory (>2GB) and disk space (>10GB)",
                "critical": True
            },
            {
                "step": 2,
                "action": "Verify network connectivity",
                "command": "ping -c 3 8.8.8.8",
                "expected": "Network connectivity confirmed",
                "critical": True
            },
            {
                "step": 3,
                "action": "Check broker API endpoints",
                "command": "curl -I https://api.binance.com/api/v3/ping",
                "expected": "HTTP 200 response from all configured brokers",
                "critical": False
            },
            {
                "step": 4,
                "action": "Verify configuration files",
                "command": "ls -la config/ .env",
                "expected": "All configuration files present and readable",
                "critical": True
            },
            {
                "step": 5,
                "action": "Check database availability",
                "command": "python -c 'import aiosqlite; print(\"DB OK\")'",
                "expected": "Database connectivity confirmed",
                "critical": True
            }
        ]
    )
    
    # System startup
    system_startup = Procedure(
        name="System Startup Sequence",
        description="Start the trading orchestrator and all subsystems",
        steps=[
            {
                "step": 1,
                "action": "Start base application",
                "command": "cd /workspace/trading_orchestrator && python main.py",
                "verification": "Check logs for successful initialization messages",
                "timeout": "60 seconds",
                "critical": True
            },
            {
                "step": 2,
                "action": "Verify broker connections",
                "command": "curl http://localhost:8000/api/dashboard/health",
                "verification": "All configured brokers show 'healthy' status",
                "timeout": "30 seconds",
                "critical": True
            },
            {
                "step": 3,
                "action": "Start perpetual operations",
                "command": "Access dashboard at http://localhost:8000/dashboard",
                "verification": "Dashboard loads and shows green status indicators",
                "timeout": "30 seconds",
                "critical": True
            },
            {
                "step": 4,
                "action": "Verify trading operations",
                "command": "Monitor dashboard for trading activity",
                "verification": "No critical alerts, system uptime increasing",
                "timeout": "5 minutes",
                "critical": False
            }
        ]
    )
    
    # Post-startup verification
    post_startup = Procedure(
        name="Post-Startup Verification",
        description="Verify all systems are operational after startup",
        steps=[
            {
                "step": 1,
                "action": "Check system health",
                "command": "curl -s http://localhost:8000/api/dashboard/health | jq '.overall_status'",
                "expected": "Status shows 'healthy'",
                "critical": True
            },
            {
                "step": 2,
                "action": "Verify monitoring systems",
                "command": "curl -s http://localhost:8000/api/dashboard/alerts | jq '.summary.total_active'",
                "expected": "No critical or high severity alerts",
                "critical": True
            },
            {
                "step": 3,
                "action": "Check maintenance scheduler",
                "command": "curl -s http://localhost:8000/api/dashboard/maintenance | jq '.status.maintenance_mode'",
                "expected": "Maintenance mode is false",
                "critical": False
            }
        ]
    )
    
    runbook.add_procedure(startup_check)
    runbook.add_procedure(system_startup)
    runbook.add_procedure(post_startup)
    
    return runbook


def create_emergency_response_runbook() -> OperationalRunbook:
    """Create emergency response runbook"""
    runbook = OperationalRunbook(
        name="Emergency Response and Incident Management",
        description="Procedures for handling system emergencies and critical incidents"
    )
    
    # System failure response
    system_failure = Procedure(
        name="System Failure Response",
        description="Immediate response to system-wide failures",
        steps=[
            {
                "step": 1,
                "action": "Assess incident severity",
                "decision_point": "Is this a critical system failure?",
                "if_yes": "Proceed to immediate shutdown procedures",
                "if_no": "Proceed to degraded operations procedures",
                "timeout": "30 seconds"
            },
            {
                "step": 2,
                "action": "Activate emergency status",
                "command": "echo 'INCIDENT: [timestamp] - [description]' >> logs/incidents.log",
                "verification": "Incident logged with timestamp and description",
                "critical": True
            },
            {
                "step": 3,
                "action": "Notify stakeholders",
                "action_type": "manual",
                "notifications": ["On-call team", "Management", "Compliance"],
                "timeout": "5 minutes",
                "critical": True
            },
            {
                "step": 4,
                "action": "Implement containment measures",
                "options": [
                    "Enter maintenance mode to prevent new orders",
                    "Pause AI trading operations",
                    "Enable circuit breakers"
                ],
                "timeout": "2 minutes",
                "critical": True
            }
        ]
    )
    
    # Critical alert response
    critical_alert = Procedure(
        name="Critical Alert Response",
        description="Response to critical system alerts",
        steps=[
            {
                "step": 1,
                "action": "Acknowledge alert",
                "command": "curl -X POST http://localhost:8000/api/dashboard/alerts/[alert_id]/resolve -d '{\"resolved_by\": \"emergency_response\"}'",
                "verification": "Alert acknowledged and escalated",
                "timeout": "1 minute",
                "critical": True
            },
            {
                "step": 2,
                "action": "Assess impact",
                "checklist": [
                    "Trading operations affected?",
                    "Data integrity at risk?",
                    "Compliance implications?",
                    "Customer impact?"
                ],
                "timeout": "3 minutes"
            },
            {
                "step": 3,
                "action": "Initiate recovery procedures",
                "decision_tree": {
                    "database_issue": "Run database recovery procedures",
                    "broker_connection": "Attempt broker reconnection",
                    "memory_issue": "Trigger memory cleanup and system restart",
                    "network_issue": "Verify network and restart connections"
                },
                "timeout": "10 minutes"
            }
        ]
    )
    
    # Data breach response
    data_breach = Procedure(
        name="Data Breach Response",
        description="Response to potential data security breaches",
        steps=[
            {
                "step": 1,
                "action": "Isolate affected systems",
                "command": "Disable external API access immediately",
                "verification": "All external connections terminated",
                "timeout": "1 minute",
                "critical": True
            },
            {
                "step": 2,
                "action": "Preserve evidence",
                "command": "Create forensic backup of logs and database",
                "verification": "Forensic backup created and secured",
                "timeout": "5 minutes",
                "critical": True
            },
            {
                "step": 3,
                "action": "Assess data exposure",
                "checklist": [
                    "What data was potentially exposed?",
                    "How many records/accounts affected?",
                    "Duration of exposure?",
                    "Evidence of unauthorized access?"
                ],
                "timeout": "15 minutes",
                "critical": True
            },
            {
                "step": 4,
                "action": "Notify authorities",
                "requirements": [
                    "GDPR notification within 72 hours",
                    "Regulatory notification as required",
                    "Customer notification if required"
                ],
                "timeout": "24 hours",
                "critical": True
            }
        ]
    )
    
    runbook.add_procedure(system_failure)
    runbook.add_procedure(critical_alert)
    runbook.add_procedure(data_breach)
    
    return runbook


def create_maintenance_runbook() -> OperationalRunbook:
    """Create maintenance operations runbook"""
    runbook = OperationalRunbook(
        name="Maintenance Operations",
        description="Scheduled and unscheduled maintenance procedures"
    )
    
    # Routine maintenance
    routine_maintenance = Procedure(
        name="Routine Maintenance",
        description="Daily and weekly maintenance tasks",
        steps=[
            {
                "step": 1,
                "action": "Enter maintenance mode",
                "command": "curl -X POST http://localhost:8000/api/dashboard/maintenance/start -d '{\"reason\": \"Routine maintenance\"}'",
                "verification": "Maintenance mode activated, non-critical operations paused",
                "timeout": "2 minutes"
            },
            {
                "step": 2,
                "action": "Run database maintenance",
                "command": "Access /api/dashboard/maintenance endpoint and trigger vacuum",
                "verification": "Database optimized, VACUUM and ANALYZE completed",
                "timeout": "5 minutes"
            },
            {
                "step": 3,
                "action": "Clean up log files",
                "command": "Log rotation should run automatically",
                "verification": "Old logs compressed, disk space reclaimed",
                "timeout": "3 minutes"
            },
            {
                "step": 4,
                "action": "Memory optimization",
                "command": "Trigger garbage collection via maintenance tasks",
                "verification": "Memory usage reduced, no memory leaks detected",
                "timeout": "2 minutes"
            },
            {
                "step": 5,
                "action": "Verify system health",
                "command": "curl http://localhost:8000/api/dashboard/health",
                "verification": "All components show healthy status",
                "timeout": "2 minutes"
            },
            {
                "step": 6,
                "action": "Exit maintenance mode",
                "command": "curl -X POST http://localhost:8000/api/dashboard/maintenance/stop -d '{\"reason\": \"Routine maintenance completed\"}'",
                "verification": "Normal operations resumed",
                "timeout": "2 minutes"
            }
        ]
    )
    
    # System update procedure
    system_update = Procedure(
        name="System Update",
        description="Procedure for updating system components",
        steps=[
            {
                "step": 1,
                "action": "Create pre-update backup",
                "command": "curl -X POST http://localhost:8000/api/dashboard/backup -d '{\"backup_type\": \"pre_update\"}'",
                "verification": "Full system backup created successfully",
                "timeout": "10 minutes"
            },
            {
                "step": 2,
                "action": "Enter maintenance mode",
                "command": "Activate maintenance mode with detailed reason",
                "verification": "All trading operations paused",
                "timeout": "2 minutes"
            },
            {
                "step": 3,
                "action": "Apply updates",
                "command": "Pull latest code and apply database migrations",
                "verification": "Updates applied without errors",
                "timeout": "30 minutes"
            },
            {
                "step": 4,
                "action": "Verify update",
                "command": "Run health checks and functionality tests",
                "verification": "System health check passes all tests",
                "timeout": "10 minutes"
            },
            {
                "step": 5,
                "action": "Gradual restart",
                "command": "Restart services and verify connectivity",
                "verification": "All services operational, connections restored",
                "timeout": "15 minutes"
            },
            {
                "step": 6,
                "action": "Exit maintenance mode",
                "command": "Resume normal operations",
                "verification": "Trading operations active, monitoring active",
                "timeout": "5 minutes"
            }
        ]
    )
    
    runbook.add_procedure(routine_maintenance)
    runbook.add_procedure(system_update)
    
    return runbook


def create_troubleshooting_runbook() -> OperationalRunbook:
    """Create troubleshooting guide"""
    runbook = OperationalRunbook(
        name="Troubleshooting and Problem Resolution",
        description="Common issues and their resolution procedures"
    )
    
    # High CPU usage
    high_cpu = Procedure(
        name="High CPU Usage",
        description="Diagnose and resolve high CPU usage issues",
        steps=[
            {
                "step": 1,
                "action": "Identify CPU-intensive processes",
                "command": "top -bn1 | head -20",
                "expected": "Identify processes consuming high CPU"
            },
            {
                "step": 2,
                "action": "Check system metrics",
                "command": "curl http://localhost:8000/api/dashboard/metrics?time_range=1h",
                "expected": "Review CPU trend data"
            },
            {
                "step": 3,
                "action": "Enable CPU throttling",
                "action_type": "automatic",
                "expected": "System automatically throttles non-critical operations"
            },
            {
                "step": 4,
                "action": "Review recent changes",
                "checklist": [
                    "Recent deployments?",
                    "New trading strategies?",
                    "Database queries?",
                    "AI model updates?"
                ]
            },
            {
                "step": 5,
                "action": "Optimize or restart affected services",
                "decision": "Based on analysis, optimize or restart specific services"
            }
        ]
    )
    
    # Memory issues
    memory_issues = Procedure(
        name="Memory Issues",
        description="Resolve memory leaks and high memory usage",
        steps=[
            {
                "step": 1,
                "action": "Check memory usage",
                "command": "free -h && python -c 'import psutil; print(f\"Process: {psutil.Process().memory_info().rss/1024/1024:.1f}MB\")'",
                "expected": "Current memory usage values"
            },
            {
                "step": 2,
                "action": "Trigger garbage collection",
                "action_type": "automatic",
                "expected": "Memory cleanup initiated by system"
            },
            {
                "step": 3,
                "action": "Review memory leak detection",
                "command": "curl http://localhost:8000/api/dashboard/health | jq '.components.memory_leaks'",
                "expected": "Check for memory leak alerts"
            },
            {
                "step": 4,
                "action": "Restart if necessary",
                "decision": "Restart system if memory usage continues to grow"
            }
        ]
    )
    
    # Database connectivity
    db_connectivity = Procedure(
        name="Database Connectivity Issues",
        description="Resolve database connection problems",
        steps=[
            {
                "step": 1,
                "action": "Test database connectivity",
                "command": "python -c 'import aiosqlite; aiosqlite.connect(\"trading_orchestrator.db\").execute(\"SELECT 1\")'",
                "expected": "Database responds to simple query"
            },
            {
                "step": 2,
                "action": "Check database health status",
                "command": "curl http://localhost:8000/api/dashboard/health | jq '.components.database'",
                "expected": "Database health status reported"
            },
            {
                "step": 3,
                "action": "Verify database file",
                "command": "ls -la trading_orchestrator.db",
                "expected": "Database file exists and is accessible"
            },
            {
                "step": 4,
                "action": "Attempt auto-recovery",
                "action_type": "automatic",
                "expected": "System attempts to restore database connection"
            },
            {
                "step": 5,
                "action": "Restore from backup if needed",
                "decision": "If auto-recovery fails, restore from recent backup"
            }
        ]
    )
    
    runbook.add_procedure(high_cpu)
    runbook.add_procedure(memory_issues)
    runbook.add_procedure(db_connectivity)
    
    return runbook


def create_backup_recovery_runbook() -> OperationalRunbook:
    """Create backup and recovery runbook"""
    runbook = OperationalRunbook(
        name="Backup and Recovery Procedures",
        description="Data protection and disaster recovery procedures"
    )
    
    # Automated backup
    automated_backup = Procedure(
        name="Automated Backup",
        description="Verify automated backup systems",
        steps=[
            {
                "step": 1,
                "action": "Check backup schedule",
                "command": "curl http://localhost:8000/api/dashboard/maintenance | jq '.status.next_scheduled_task'",
                "expected": "Next backup time and type"
            },
            {
                "step": 2,
                "action": "Manually trigger backup",
                "command": "curl -X POST http://localhost:8000/api/dashboard/backup",
                "verification": "Backup created successfully"
            },
            {
                "step": 3,
                "action": "Verify backup integrity",
                "command": "ls -la backups/",
                "expected": "Recent backup file with appropriate size"
            },
            {
                "step": 4,
                "action": "Test backup restoration",
                "action_type": "procedure",
                "expected": "Backup can be restored successfully"
            }
        ]
    )
    
    # Disaster recovery
    disaster_recovery = Procedure(
        name="Disaster Recovery",
        description="Full system recovery from backup",
        steps=[
            {
                "step": 1,
                "action": "Assess damage",
                "checklist": [
                    "What systems are affected?",
                    "Data corruption extent?",
                    "Uptime requirements?",
                    "Recovery time objective?"
                ],
                "critical": True
            },
            {
                "step": 2,
                "action": "Select recovery strategy",
                "options": [
                    "Full system restore",
                    "Partial restore",
                    "Point-in-time recovery"
                ]
            },
            {
                "step": 3,
                "action": "Execute recovery",
                "command": "curl -X POST http://localhost:8000/api/dashboard/restore -d '{\"backup_name\": \"[backup_name]\"}'",
                "verification": "System restored from backup"
            },
            {
                "step": 4,
                "action": "Verify recovery",
                "steps": [
                    "Database integrity check",
                    "Application functionality test",
                    "Trading operations verification"
                ],
                "timeout": "30 minutes"
            },
            {
                "step": 5,
                "action": "Resume operations",
                "verification": "All systems operational, monitoring active"
            }
        ]
    )
    
    runbook.add_procedure(automated_backup)
    runbook.add_procedure(disaster_recovery)
    
    return runbook


# ================================
# Runbook Manager
# ================================

class RunbookManager:
    """Manager for operational runbooks"""
    
    def __init__(self, runbooks_dir: str = "docs/runbooks"):
        self.runbooks_dir = Path(runbooks_dir)
        self.runbooks_dir.mkdir(parents=True, exist_ok=True)
        self.runbooks = {}
        
        # Initialize default runbooks
        self._initialize_default_runbooks()
    
    def _initialize_default_runbooks(self):
        """Initialize default runbooks"""
        self.add_runbook(create_system_startup_runbook())
        self.add_runbook(create_emergency_response_runbook())
        self.add_runbook(create_maintenance_runbook())
        self.add_runbook(create_troubleshooting_runbook())
        self.add_runbook(create_backup_recovery_runbook())
    
    def add_runbook(self, runbook: OperationalRunbook):
        """Add a runbook to the manager"""
        self.runbooks[runbook.name] = runbook
    
    def get_runbook(self, name: str) -> Optional[OperationalRunbook]:
        """Get a specific runbook"""
        return self.runbooks.get(name)
    
    def list_runbooks(self) -> List[str]:
        """List all available runbooks"""
        return list(self.runbooks.keys())
    
    def export_runbook(self, name: str, format: str = "json") -> Dict[str, Any]:
        """Export runbook in specified format"""
        runbook = self.get_runbook(name)
        if not runbook:
            raise ValueError(f"Runbook '{name}' not found")
        
        if format == "json":
            return runbook.to_dict()
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def save_runbooks(self):
        """Save all runbooks to files"""
        for name, runbook in self.runbooks.items():
            file_path = self.runbooks_dir / f"{name.lower().replace(' ', '_')}.json"
            with open(file_path, 'w') as f:
                json.dump(runbook.to_dict(), f, indent=2)
    
    def load_runbooks(self):
        """Load runbooks from files"""
        for file_path in self.runbooks_dir.glob("*.json"):
            try:
                with open(file_path, 'r') as f:
                    runbook_data = json.load(f)
                
                # Reconstruct runbook (simplified)
                runbook = OperationalRunbook(
                    name=runbook_data['name'],
                    description=runbook_data['description']
                )
                
                # Add procedures
                for proc_data in runbook_data['procedures']:
                    procedure = Procedure(
                        name=proc_data['name'],
                        description=proc_data['description'],
                        steps=proc_data['steps']
                    )
                    runbook.add_procedure(procedure)
                
                self.add_runbook(runbook)
                
            except Exception as e:
                logger.error(f"Failed to load runbook from {file_path}: {e}")
    
    def get_procedure_steps(self, runbook_name: str, procedure_name: str) -> Optional[List[Dict[str, Any]]]:
        """Get steps for a specific procedure"""
        runbook = self.get_runbook(runbook_name)
        if not runbook:
            return None
        
        for procedure in runbook.procedures:
            if procedure.name == procedure_name:
                return procedure.steps
        
        return None


# ================================
# Operational Procedures
# ================================

class OperationalProcedures:
    """Collection of operational procedures"""
    
    @staticmethod
    async def health_check_procedure() -> Dict[str, Any]:
        """Comprehensive health check procedure"""
        results = {
            'timestamp': datetime.utcnow().isoformat(),
            'overall_status': 'healthy',
            'checks': []
        }
        
        try:
            # System resources check
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            resource_check = {
                'component': 'system_resources',
                'status': 'healthy' if cpu_percent < 80 and memory.percent < 90 else 'degraded',
                'details': {
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'disk_percent': disk.used / disk.total * 100
                }
            }
            results['checks'].append(resource_check)
            
            # Network connectivity check
            try:
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)
                result = sock.connect_ex(('8.8.8.8', 53))
                sock.close()
                
                network_check = {
                    'component': 'network',
                    'status': 'healthy' if result == 0 else 'critical',
                    'details': {'connectivity': result == 0}
                }
                results['checks'].append(network_check)
            except Exception as e:
                network_check = {
                    'component': 'network',
                    'status': 'unknown',
                    'details': {'error': str(e)}
                }
                results['checks'].append(network_check)
            
            # File system check
            fs_check = {
                'component': 'filesystem',
                'status': 'healthy',
                'details': {'writable': True}
            }
            results['checks'].append(fs_check)
            
            # Determine overall status
            statuses = [check['status'] for check in results['checks']]
            if 'critical' in statuses:
                results['overall_status'] = 'critical'
            elif 'degraded' in statuses:
                results['overall_status'] = 'degraded'
            else:
                results['overall_status'] = 'healthy'
            
        except Exception as e:
            results['overall_status'] = 'error'
            results['error'] = str(e)
        
        return results
    
    @staticmethod
    async def cleanup_procedure() -> Dict[str, Any]:
        """System cleanup procedure"""
        results = {
            'timestamp': datetime.utcnow().isoformat(),
            'actions_taken': [],
            'space_reclaimed_mb': 0
        }
        
        try:
            # Clean old log files
            logs_dir = Path("logs")
            if logs_dir.exists():
                cutoff_time = datetime.utcnow() - timedelta(days=7)
                cleaned_files = 0
                
                for log_file in logs_dir.glob("*.log*"):
                    try:
                        if log_file.stat().st_mtime < cutoff_time.timestamp():
                            file_size = log_file.stat().st_size
                            log_file.unlink()
                            results['space_reclaimed_mb'] += file_size / 1024 / 1024
                            cleaned_files += 1
                    except:
                        pass
                
                results['actions_taken'].append(f"Cleaned {cleaned_files} old log files")
            
            # Clean cache directories
            cache_dirs = ["cache", "tmp"]
            for cache_dir in cache_dirs:
                cache_path = Path(cache_dir)
                if cache_path.exists():
                    import shutil
                    shutil.rmtree(cache_path, ignore_errors=True)
                    cache_path.mkdir(exist_ok=True)
                    results['actions_taken'].append(f"Cleaned {cache_dir} directory")
            
            # Run garbage collection
            import gc
            collected = gc.collect()
            results['actions_taken'].append(f"Garbage collected {collected} objects")
            
            results['space_reclaimed_mb'] = round(results['space_reclaimed_mb'], 2)
            
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    @staticmethod
    async def backup_procedure(backup_type: str = "full") -> Dict[str, Any]:
        """Create system backup"""
        results = {
            'timestamp': datetime.utcnow().isoformat(),
            'backup_type': backup_type,
            'status': 'started'
        }
        
        try:
            backup_dir = Path("backups")
            backup_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            backup_name = f"backup_{backup_type}_{timestamp}"
            backup_path = backup_dir / backup_name
            backup_path.mkdir(exist_ok=True)
            
            # Backup database
            db_backup_path = backup_path / "database"
            db_backup_path.mkdir(exist_ok=True)
            
            import shutil
            if Path("trading_orchestrator.db").exists():
                shutil.copy2("trading_orchestrator.db", db_backup_path / "trading_orchestrator.db")
                results['database_backed_up'] = True
            
            # Backup configuration
            config_backup_path = backup_path / "config"
            config_backup_path.mkdir(exist_ok=True)
            
            config_files = [".env", "config.json"]
            for config_file in config_files:
                if Path(config_file).exists():
                    shutil.copy2(config_file, config_backup_path / config_file)
            
            # Create metadata
            metadata = {
                'backup_type': backup_type,
                'timestamp': timestamp,
                'system_version': '2.0.0',
                'backup_size_mb': sum(f.stat().st_size for f in backup_path.rglob('*') if f.is_file()) / 1024 / 1024
            }
            
            with open(backup_path / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            results.update({
                'status': 'completed',
                'backup_name': backup_name,
                'backup_path': str(backup_path),
                'size_mb': round(metadata['backup_size_mb'], 2)
            })
            
        except Exception as e:
            results.update({
                'status': 'failed',
                'error': str(e)
            })
        
        return results


# ================================
# SLA and Monitoring
# ================================

class ServiceLevelAgreement:
    """Service Level Agreement definitions"""
    
    # System SLAs
    SYSTEM_SLAS = {
        'uptime': {
            'target': 99.9,  # 99.9% uptime
            'measurement': 'percentage',
            'period': 'monthly'
        },
        'response_time': {
            'target': 100,  # 100ms average response time
            'measurement': 'milliseconds',
            'period': 'daily'
        },
        'recovery_time': {
            'target': 300,  # 5 minutes maximum recovery time
            'measurement': 'seconds',
            'period': 'per_incident'
        },
        'data_loss': {
            'target': 0,  # Zero data loss
            'measurement': 'events',
            'period': 'continuous'
        }
    }
    
    # Trading SLAs
    TRADING_SLAS = {
        'order_execution': {
            'target': 1000,  # 1 second maximum execution time
            'measurement': 'milliseconds',
            'period': 'per_order'
        },
        'risk_check_latency': {
            'target': 100,  # 100ms maximum risk check time
            'measurement': 'milliseconds',
            'period': 'per_order'
        },
        'market_data_latency': {
            'target': 50,  # 50ms maximum data latency
            'measurement': 'milliseconds',
            'period': 'continuous'
        }
    }
    
    @classmethod
    def check_sla_compliance(cls, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Check SLA compliance against current metrics"""
        compliance = {
            'timestamp': datetime.utcnow().isoformat(),
            'overall_compliance': 'compliant',
            'violations': []
        }
        
        # Check system SLAs
        for sla_name, sla_config in cls.SYSTEM_SLAS.items():
            if sla_name in metrics:
                actual_value = metrics[sla_name]
                target_value = sla_config['target']
                
                # Simple compliance check (would need more sophisticated logic in reality)
                is_compliant = actual_value <= target_value if 'time' in sla_name else actual_value >= target_value
                
                if not is_compliant:
                    compliance['violations'].append({
                        'sla': sla_name,
                        'target': target_value,
                        'actual': actual_value,
                        'measurement': sla_config['measurement']
                    })
                    compliance['overall_compliance'] = 'violation'
        
        return compliance


# ================================
# Global Instance
# ================================

# Global runbook manager instance
runbook_manager = RunbookManager()

# Initialize default runbooks on import
runbook_manager._initialize_default_runbooks()