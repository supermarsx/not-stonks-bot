"""
System Integration Manager
Coordinates all system components and manages cross-component communication.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import time
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComponentStatus(Enum):
    """Component status enumeration."""
    INITIALIZING = "initializing"
    RUNNING = "running"
    ERROR = "error"
    STOPPED = "stopped"
    DEGRADED = "degraded"


@dataclass
class ComponentInfo:
    """Component information container."""
    name: str
    version: str
    status: ComponentStatus
    dependencies: List[str]
    initialized_at: datetime
    last_health_check: Optional[datetime] = None
    error_count: int = 0
    metrics: Dict[str, Any] = None


class IntegrationManager:
    """Manages system integration and component coordination."""
    
    def __init__(self):
        self.components: Dict[str, ComponentInfo] = {}
        self.message_bus: Dict[str, List[Any]] = {}
        self.dependency_graph: Dict[str, List[str]] = {}
        self.event_handlers: Dict[str, List] = {}
        self.integration_config = self._load_integration_config()
        self.heartbeat_interval = 30
        self.health_check_interval = 60
        
    def _load_integration_config(self) -> Dict[str, Any]:
        """Load integration configuration."""
        try:
            with open('config/integration.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning("Integration config not found, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default integration configuration."""
        return {
            "components": {
                "crawler_manager": {
                    "dependencies": [],
                    "priority": 1
                },
                "trading_orchestrator": {
                    "dependencies": ["crawler_manager"],
                    "priority": 2
                },
                "risk_manager": {
                    "dependencies": ["trading_orchestrator"],
                    "priority": 3
                },
                "ui_components": {
                    "dependencies": ["trading_orchestrator"],
                    "priority": 4
                }
            },
            "message_timeout": 30,
            "max_retries": 3,
            "circuit_breaker_threshold": 5
        }
    
    async def register_component(self, name: str, version: str, 
                                dependencies: List[str] = None) -> bool:
        """Register a new system component."""
        try:
            component_info = ComponentInfo(
                name=name,
                version=version,
                status=ComponentStatus.INITIALIZING,
                dependencies=dependencies or [],
                initialized_at=datetime.now(),
                metrics={}
            )
            
            self.components[name] = component_info
            self.message_bus[name] = []
            
            # Update dependency graph
            if dependencies:
                for dep in dependencies:
                    if dep not in self.dependency_graph:
                        self.dependency_graph[dep] = []
                    self.dependency_graph[dep].append(name)
            
            logger.info(f"Component {name} v{version} registered successfully")
            await self._validate_dependencies(name)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to register component {name}: {str(e)}")
            return False
    
    async def _validate_dependencies(self, component_name: str) -> bool:
        """Validate component dependencies."""
        component = self.components.get(component_name)
        if not component:
            return False
            
        for dep_name in component.dependencies:
            if dep_name not in self.components:
                logger.error(f"Missing dependency {dep_name} for {component_name}")
                component.status = ComponentStatus.ERROR
                return False
                
            dep_status = self.components[dep_name].status
            if dep_status not in [ComponentStatus.RUNNING, ComponentStatus.DEGRADED]:
                logger.warning(f"Dependency {dep_name} not ready for {component_name}")
                return False
                
        return True
    
    async def initialize_component(self, component_name: str) -> bool:
        """Initialize a specific component."""
        try:
            if component_name not in self.components:
                logger.error(f"Component {component_name} not registered")
                return False
            
            component = self.components[component_name]
            
            # Check dependencies
            if not await self._validate_dependencies(component_name):
                logger.error(f"Dependencies not met for {component_name}")
                return False
            
            # Initialize component
            success = await self._perform_component_init(component_name)
            
            if success:
                component.status = ComponentStatus.RUNNING
                component.last_health_check = datetime.now()
                logger.info(f"Component {component_name} initialized successfully")
                
                # Notify dependent components
                await self._notify_dependents(component_name)
            else:
                component.status = ComponentStatus.ERROR
                component.error_count += 1
                logger.error(f"Failed to initialize component {component_name}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error initializing component {component_name}: {str(e)}")
            return False
    
    async def _perform_component_init(self, component_name: str) -> bool:
        """Perform actual component initialization."""
        # This would call the actual component initialization
        # For now, simulate successful initialization
        await asyncio.sleep(0.1)
        return True
    
    async def _notify_dependents(self, component_name: str):
        """Notify components that depend on this component."""
        if component_name in self.dependency_graph:
            for dependent in self.dependency_graph[component_name]:
                await self.send_message(component_name, dependent, 
                                      "dependency_ready", {})
    
    async def send_message(self, sender: str, receiver: str, 
                          message_type: str, data: Dict[str, Any]) -> bool:
        """Send message between components."""
        try:
            message = {
                "sender": sender,
                "receiver": receiver,
                "type": message_type,
                "data": data,
                "timestamp": datetime.now().isoformat(),
                "id": f"{sender}-{receiver}-{int(time.time() * 1000)}"
            }
            
            if receiver in self.message_bus:
                self.message_bus[receiver].append(message)
                logger.debug(f"Message sent from {sender} to {receiver}: {message_type}")
                return True
            else:
                logger.error(f"Receiver {receiver} not found")
                return False
                
        except Exception as e:
            logger.error(f"Failed to send message: {str(e)}")
            return False
    
    def get_messages(self, component_name: str, message_type: str = None) -> List[Dict[str, Any]]:
        """Get messages for a component."""
        if component_name not in self.message_bus:
            return []
        
        messages = self.message_bus[component_name]
        
        if message_type:
            messages = [msg for msg in messages if msg["type"] == message_type]
        
        # Clear retrieved messages
        self.message_bus[component_name] = []
        
        return messages
    
    async def health_check_component(self, component_name: str) -> Dict[str, Any]:
        """Perform health check on a component."""
        try:
            if component_name not in self.components:
                return {"status": "not_found", "error": "Component not registered"}
            
            component = self.components[component_name]
            
            # Simulate health check
            health_status = await self._perform_health_check(component_name)
            
            component.last_health_check = datetime.now()
            
            if health_status["status"] == "healthy":
                if component.status == ComponentStatus.ERROR:
                    component.status = ComponentStatus.RUNNING
                    component.error_count = 0
            else:
                component.status = ComponentStatus.ERROR
                component.error_count += 1
            
            return health_status
            
        except Exception as e:
            logger.error(f"Health check failed for {component_name}: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    async def _perform_health_check(self, component_name: str) -> Dict[str, Any]:
        """Perform actual health check."""
        # Simulate health check - in real implementation, this would check
        # the actual component health
        import random
        return {
            "status": random.choice(["healthy", "degraded", "error"]),
            "cpu_usage": random.uniform(10, 80),
            "memory_usage": random.uniform(20, 70),
            "response_time": random.uniform(0.01, 0.5),
            "timestamp": datetime.now().isoformat()
        }
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        status = {
            "timestamp": datetime.now().isoformat(),
            "total_components": len(self.components),
            "running_components": 0,
            "error_components": 0,
            "components": {}
        }
        
        for name, component in self.components.items():
            component_status = {
                "name": name,
                "version": component.version,
                "status": component.status.value,
                "uptime": (datetime.now() - component.initialized_at).total_seconds(),
                "error_count": component.error_count,
                "last_health_check": component.last_health_check.isoformat() 
                    if component.last_health_check else None
            }
            
            status["components"][name] = component_status
            
            if component.status == ComponentStatus.RUNNING:
                status["running_components"] += 1
            elif component.status == ComponentStatus.ERROR:
                status["error_components"] += 1
        
        status["system_health"] = self._calculate_system_health(status)
        
        return status
    
    def _calculate_system_health(self, status: Dict[str, Any]) -> str:
        """Calculate overall system health."""
        total = status["total_components"]
        running = status["running_components"]
        
        if total == 0:
            return "unknown"
        
        health_ratio = running / total
        
        if health_ratio == 1.0:
            return "healthy"
        elif health_ratio >= 0.8:
            return "degraded"
        else:
            return "critical"
    
    async def restart_component(self, component_name: str) -> bool:
        """Restart a component."""
        try:
            if component_name not in self.components:
                return False
            
            logger.info(f"Restarting component {component_name}")
            
            # Stop component
            self.components[component_name].status = ComponentStatus.STOPPED
            
            # Wait for graceful shutdown
            await asyncio.sleep(2)
            
            # Restart component
            success = await self.initialize_component(component_name)
            
            if success:
                logger.info(f"Component {component_name} restarted successfully")
            else:
                logger.error(f"Failed to restart component {component_name}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error restarting component {component_name}: {str(e)}")
            return False
    
    async def shutdown(self):
        """Gracefully shutdown all components."""
        logger.info("Shutting down system components...")
        
        # Stop components in reverse dependency order
        sorted_components = self._get_stop_order()
        
        for component_name in sorted_components:
            if component_name in self.components:
                self.components[component_name].status = ComponentStatus.STOPPED
                logger.info(f"Stopped component {component_name}")
        
        logger.info("System shutdown complete")
    
    def _get_stop_order(self) -> List[str]:
        """Get components in proper stop order (reverse dependency)."""
        # Simple implementation - would need proper topological sort in production
        return list(self.components.keys())


# Global integration manager instance
integration_manager = IntegrationManager()