"""
Perpetual Operations Dashboard
=============================

Real-time monitoring dashboard for perpetual trading operations.
Provides comprehensive visibility into system health, performance, and maintenance status.

Features:
- Real-time system metrics visualization
- Health status monitoring
- Alert dashboard with escalation tracking
- Performance analytics and trends
- Maintenance schedule and status
- Recovery operation logs
- Resource utilization charts
- Connection pool monitoring
- Memory leak detection plots

Author: Trading Orchestrator System
Version: 2.0.0
Date: 2025-11-06
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
import aiohttp
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from loguru import logger

from .perpetual_manager import (
    PerpetualOperationsManager, SystemMetrics, HealthCheckResult, 
    AlertInfo, MaintenanceTask, AlertManager
)


# ================================
# Data Models
# ================================

class DashboardMetrics(BaseModel):
    """Dashboard metrics data model"""
    timestamp: str
    system_metrics: SystemMetrics
    health_status: Dict[str, HealthCheckResult]
    active_alerts: List[AlertInfo]
    maintenance_status: Dict[str, Any]
    performance_stats: Dict[str, Any]
    uptime_seconds: float


class WebSocketMessage(BaseModel):
    """WebSocket message model"""
    type: str  # 'metrics', 'health', 'alert', 'maintenance'
    data: Dict[str, Any]
    timestamp: str


# ================================
# Dashboard API Endpoints
# ================================

def setup_dashboard_routes(app: FastAPI, perpetual_manager: PerpetualOperationsManager):
    """Setup dashboard API routes"""
    
    @app.get("/api/dashboard/overview")
    async def get_dashboard_overview():
        """Get dashboard overview data"""
        try:
            system_status = perpetual_manager.get_system_status()
            
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'status': 'healthy' if system_status['is_running'] else 'stopped',
                'uptime_hours': system_status['uptime_seconds'] / 3600,
                'active_alerts': system_status['active_alerts'],
                'health_components': len([h for h in system_status['health_status'].values() if h.get('healthy', False)]),
                'total_components': len(system_status['health_status']),
                'maintenance_mode': system_status['maintenance_status']['maintenance_mode']
            }
            
        except Exception as e:
            logger.error(f"Dashboard overview error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/dashboard/metrics")
    async def get_metrics(time_range: str = Query("1h", description="Time range: 1h, 6h, 24h, 7d")):
        """Get system metrics for specified time range"""
        try:
            # Parse time range
            time_ranges = {
                "1h": timedelta(hours=1),
                "6h": timedelta(hours=6),
                "24h": timedelta(hours=24),
                "7d": timedelta(days=7)
            }
            
            if time_range not in time_ranges:
                raise HTTPException(status_code=400, detail="Invalid time range")
            
            cutoff_time = datetime.utcnow() - time_ranges[time_range]
            
            # Get metrics from history
            metrics_history = []
            for metrics in perpetual_manager.monitor.metrics_history:
                if datetime.fromisoformat(metrics.timestamp) >= cutoff_time:
                    metrics_history.append(metrics.to_dict())
            
            return {
                'time_range': time_range,
                'metrics': metrics_history,
                'count': len(metrics_history)
            }
            
        except Exception as e:
            logger.error(f"Metrics API error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/dashboard/health")
    async def get_health_status():
        """Get detailed health status"""
        try:
            health_status = perpetual_manager.health_checker.health_checks
            
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'components': {
                    name: result.to_dict() 
                    for name, result in health_status.items()
                },
                'overall_status': 'healthy' if all(
                    result.is_healthy() for result in health_status.values()
                ) else 'degraded'
            }
            
        except Exception as e:
            logger.error(f"Health status API error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/dashboard/alerts")
    async def get_alerts(
        severity: Optional[str] = Query(None, description="Filter by severity"),
        resolved: Optional[bool] = Query(False, description="Include resolved alerts"),
        limit: int = Query(50, description="Maximum number of alerts to return")
    ):
        """Get alerts with filtering options"""
        try:
            active_alerts = AlertManager.get_active_alerts()
            all_alerts = active_alerts + AlertManager.alert_history
            
            # Apply filters
            filtered_alerts = []
            for alert in all_alerts:
                if severity and alert.severity != severity:
                    continue
                if not resolved and alert.resolved:
                    continue
                filtered_alerts.append(alert.to_dict())
            
            # Limit results
            filtered_alerts = filtered_alerts[-limit:]
            
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'alerts': filtered_alerts,
                'summary': AlertManager.get_alert_summary()
            }
            
        except Exception as e:
            logger.error(f"Alerts API error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/dashboard/maintenance")
    async def get_maintenance_status():
        """Get maintenance status and schedule"""
        try:
            maintenance_status = perpetual_manager.maintenance_manager.get_maintenance_status()
            
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'status': maintenance_status
            }
            
        except Exception as e:
            logger.error(f"Maintenance API error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/dashboard/performance")
    async def get_performance_stats():
        """Get performance statistics"""
        try:
            system_status = perpetual_manager.get_system_status()
            
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'performance_stats': system_status['performance_stats'],
                'recovery_stats': system_status['recovery_stats'],
                'latest_metrics': system_status['latest_metrics']
            }
            
        except Exception as e:
            logger.error(f"Performance API error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/dashboard/maintenance/start")
    async def start_maintenance(reason: str):
        """Start maintenance mode"""
        try:
            await perpetual_manager.enter_maintenance_mode(reason)
            return {'status': 'success', 'message': 'Maintenance mode started'}
            
        except Exception as e:
            logger.error(f"Start maintenance error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/dashboard/maintenance/stop")
    async def stop_maintenance(reason: str):
        """Stop maintenance mode"""
        try:
            await perpetual_manager.exit_maintenance_mode(reason)
            return {'status': 'success', 'message': 'Maintenance mode stopped'}
            
        except Exception as e:
            logger.error(f"Stop maintenance error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/dashboard/backup")
    async def create_backup(backup_type: str = "full"):
        """Create system backup"""
        try:
            success = await perpetual_manager.create_backup(backup_type)
            if success:
                return {'status': 'success', 'message': f'{backup_type} backup created'}
            else:
                raise HTTPException(status_code=500, detail='Backup creation failed')
            
        except Exception as e:
            logger.error(f"Backup creation error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/dashboard/backups")
    async def list_backups():
        """List available backups"""
        try:
            backup_dir = Path("backups")
            if not backup_dir.exists():
                return {'backups': []}
            
            backups = []
            for backup_path in backup_dir.iterdir():
                if backup_path.is_dir():
                    metadata_file = backup_path / "metadata.json"
                    if metadata_file.exists():
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        backups.append({
                            'name': backup_path.name,
                            'timestamp': metadata.get('timestamp'),
                            'type': metadata.get('backup_type'),
                            'size_mb': metadata.get('backup_size_mb'),
                            'system_version': metadata.get('system_version')
                        })
            
            backups.sort(key=lambda x: x['timestamp'], reverse=True)
            
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'backups': backups
            }
            
        except Exception as e:
            logger.error(f"List backups error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/dashboard/alerts/{alert_id}/resolve")
    async def resolve_alert(alert_id: str, resolved_by: str = "manual"):
        """Resolve an alert"""
        try:
            AlertManager.resolve_alert(alert_id, resolved_by)
            return {'status': 'success', 'message': 'Alert resolved'}
            
        except Exception as e:
            logger.error(f"Resolve alert error: {e}")
            raise HTTPException(status_code=500, detail=str(e))


# ================================
# WebSocket Management
# ================================

class DashboardWebSocketManager:
    """WebSocket manager for real-time dashboard updates"""
    
    def __init__(self):
        self.connections: Dict[str, WebSocket] = {}
        self.connection_ids = 0
    
    async def connect(self, websocket: WebSocket, client_id: str = None):
        """Accept new WebSocket connection"""
        await websocket.accept()
        
        if not client_id:
            client_id = f"client_{self.connection_ids}"
            self.connection_ids += 1
        
        self.connections[client_id] = websocket
        logger.info(f"Dashboard WebSocket connected: {client_id}")
        
        # Send initial data
        try:
            initial_data = WebSocketMessage(
                type="connection",
                data={"client_id": client_id, "status": "connected"},
                timestamp=datetime.utcnow().isoformat()
            )
            await websocket.send_text(initial_data.model_dump_json())
        except Exception as e:
            logger.error(f"Error sending initial WebSocket data: {e}")
    
    def disconnect(self, client_id: str):
        """Remove WebSocket connection"""
        if client_id in self.connections:
            del self.connections[client_id]
            logger.info(f"Dashboard WebSocket disconnected: {client_id}")
    
    async def broadcast_message(self, message: WebSocketMessage):
        """Broadcast message to all connected clients"""
        if not self.connections:
            return
        
        message_text = message.model_dump_json()
        disconnected_clients = []
        
        for client_id, websocket in self.connections.items():
            try:
                await websocket.send_text(message_text)
            except Exception as e:
                logger.error(f"Error sending WebSocket message to {client_id}: {e}")
                disconnected_clients.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            self.disconnect(client_id)
    
    async def send_to_client(self, client_id: str, message: WebSocketMessage):
        """Send message to specific client"""
        if client_id not in self.connections:
            return
        
        try:
            await self.connections[client_id].send_text(message.model_dump_json())
        except Exception as e:
            logger.error(f"Error sending WebSocket message to {client_id}: {e}")
            self.disconnect(client_id)


def setup_websocket_routes(app: FastAPI, perpetual_manager: PerpetualOperationsManager, ws_manager: DashboardWebSocketManager):
    """Setup WebSocket routes for real-time updates"""
    
    @app.websocket("/ws/dashboard")
    async def dashboard_websocket(websocket: WebSocket, client_id: str = Query(None)):
        """WebSocket endpoint for real-time dashboard updates"""
        await ws_manager.connect(websocket, client_id)
        
        try:
            while True:
                # Receive messages from client
                data = await websocket.receive_text()
                
                # Echo back or process as needed
                response = WebSocketMessage(
                    type="echo",
                    data={"original": data, "timestamp": datetime.utcnow().isoformat()},
                    timestamp=datetime.utcnow().isoformat()
                )
                await ws_manager.send_to_client(client_id, response)
                
        except WebSocketDisconnect:
            ws_manager.disconnect(client_id)
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            ws_manager.disconnect(client_id)


# ================================
# Background Tasks
# ================================

async def broadcast_dashboard_updates(perpetual_manager: PerpetualOperationsManager, ws_manager: DashboardWebSocketManager):
    """Background task to broadcast dashboard updates"""
    while True:
        try:
            await asyncio.sleep(5)  # Update every 5 seconds
            
            # Collect latest data
            system_status = perpetual_manager.get_system_status()
            
            # Create dashboard metrics
            dashboard_metrics = DashboardMetrics(
                timestamp=datetime.utcnow().isoformat(),
                system_metrics=SystemMetrics(**system_status['latest_metrics']) if system_status['latest_metrics'] else None,
                health_status=system_status['health_status'],
                active_alerts=[AlertInfo(**alert) for alert in system_status['active_alerts']],
                maintenance_status=system_status['maintenance_status'],
                performance_stats=system_status['performance_stats'],
                uptime_seconds=system_status['uptime_seconds']
            )
            
            # Broadcast updates
            update_message = WebSocketMessage(
                type="metrics_update",
                data=dashboard_metrics.model_dump(),
                timestamp=datetime.utcnow().isoformat()
            )
            
            await ws_manager.broadcast_message(update_message)
            
            # Send individual updates for different types
            if system_status['health_status']:
                health_message = WebSocketMessage(
                    type="health_update",
                    data=system_status['health_status'],
                    timestamp=datetime.utcnow().isoformat()
                )
                await ws_manager.broadcast_message(health_message)
            
            if system_status['active_alerts']:
                alert_message = WebSocketMessage(
                    type="alert_update",
                    data=system_status['active_alerts'],
                    timestamp=datetime.utcnow().isoformat()
                )
                await ws_manager.broadcast_message(alert_message)
            
        except Exception as e:
            logger.error(f"Dashboard broadcast error: {e}")
            await asyncio.sleep(5)


# ================================
# Dashboard HTML/Frontend
# ================================

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Perpetual Trading Operations Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        .dashboard-card {
            background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%);
            border: 1px solid #3b82f6;
            border-radius: 8px;
            padding: 1.5rem;
            margin: 0.5rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        .status-indicator {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 8px;
        }
        .status-healthy { background-color: #10b981; }
        .status-degraded { background-color: #f59e0b; }
        .status-critical { background-color: #ef4444; }
        .status-unknown { background-color: #6b7280; }
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            color: #60a5fa;
        }
        .metric-label {
            color: #9ca3af;
            font-size: 0.875rem;
        }
        .alert-critical { border-left: 4px solid #ef4444; }
        .alert-high { border-left: 4px solid #f59e0b; }
        .alert-medium { border-left: 4px solid #eab308; }
        .alert-low { border-left: 4px solid #10b981; }
    </style>
</head>
<body class="bg-gray-900 text-white min-h-screen">
    <!-- Header -->
    <header class="bg-gray-800 border-b border-gray-700 p-4">
        <div class="flex justify-between items-center">
            <div class="flex items-center">
                <i class="fas fa-chart-line text-blue-400 text-2xl mr-3"></i>
                <h1 class="text-2xl font-bold">Perpetual Trading Operations</h1>
            </div>
            <div class="flex items-center space-x-4">
                <div id="connection-status" class="flex items-center">
                    <span class="status-indicator status-healthy"></span>
                    <span>Connected</span>
                </div>
                <div id="system-status" class="flex items-center">
                    <span class="status-indicator status-healthy"></span>
                    <span>All Systems Operational</span>
                </div>
            </div>
        </div>
    </header>

    <!-- Main Dashboard -->
    <main class="container mx-auto p-4">
        <!-- Key Metrics Row -->
        <div class="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
            <div class="dashboard-card">
                <div class="metric-value" id="uptime-value">0h</div>
                <div class="metric-label">System Uptime</div>
            </div>
            <div class="dashboard-card">
                <div class="metric-value" id="cpu-value">0%</div>
                <div class="metric-label">CPU Usage</div>
            </div>
            <div class="dashboard-card">
                <div class="metric-value" id="memory-value">0MB</div>
                <div class="metric-label">Memory Usage</div>
            </div>
            <div class="dashboard-card">
                <div class="metric-value" id="alerts-count">0</div>
                <div class="metric-label">Active Alerts</div>
            </div>
        </div>

        <!-- Charts Row -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
            <div class="dashboard-card">
                <h3 class="text-lg font-semibold mb-4">System Performance</h3>
                <canvas id="performance-chart" width="400" height="200"></canvas>
            </div>
            <div class="dashboard-card">
                <h3 class="text-lg font-semibold mb-4">Resource Utilization</h3>
                <canvas id="resources-chart" width="400" height="200"></canvas>
            </div>
        </div>

        <!-- Health and Alerts Row -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <!-- System Health -->
            <div class="dashboard-card">
                <h3 class="text-lg font-semibold mb-4">System Health</h3>
                <div id="health-components">
                    <!-- Health components will be populated here -->
                </div>
            </div>

            <!-- Active Alerts -->
            <div class="dashboard-card">
                <h3 class="text-lg font-semibold mb-4">Active Alerts</h3>
                <div id="active-alerts">
                    <!-- Alerts will be populated here -->
                </div>
            </div>
        </div>

        <!-- Maintenance and Actions -->
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-6 mt-6">
            <div class="dashboard-card">
                <h3 class="text-lg font-semibold mb-4">Maintenance Status</h3>
                <div id="maintenance-status">
                    <!-- Maintenance status will be populated here -->
                </div>
                <div class="mt-4 space-y-2">
                    <button onclick="startMaintenance()" class="bg-yellow-600 hover:bg-yellow-700 px-4 py-2 rounded text-sm">
                        Start Maintenance
                    </button>
                    <button onclick="stopMaintenance()" class="bg-green-600 hover:bg-green-700 px-4 py-2 rounded text-sm">
                        Stop Maintenance
                    </button>
                </div>
            </div>

            <div class="dashboard-card">
                <h3 class="text-lg font-semibold mb-4">Quick Actions</h3>
                <div class="space-y-2">
                    <button onclick="createBackup()" class="bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded text-sm w-full">
                        <i class="fas fa-download mr-2"></i>Create Backup
                    </button>
                    <button onclick="resolveAllAlerts()" class="bg-green-600 hover:bg-green-700 px-4 py-2 rounded text-sm w-full">
                        <i class="fas fa-check mr-2"></i>Resolve All Alerts
                    </button>
                    <button onclick="refreshData()" class="bg-gray-600 hover:bg-gray-700 px-4 py-2 rounded text-sm w-full">
                        <i class="fas fa-sync mr-2"></i>Refresh Data
                    </button>
                </div>
            </div>

            <div class="dashboard-card">
                <h3 class="text-lg font-semibold mb-4">Performance Stats</h3>
                <div id="performance-stats">
                    <!-- Performance stats will be populated here -->
                </div>
            </div>
        </div>
    </main>

    <script>
        // WebSocket connection
        let ws = null;
        let reconnectAttempts = 0;
        const maxReconnectAttempts = 5;
        
        // Charts
        let performanceChart = null;
        let resourcesChart = null;
        
        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', function() {
            initializeCharts();
            connectWebSocket();
            loadInitialData();
        });
        
        function initializeCharts() {
            // Performance Chart
            const performanceCtx = document.getElementById('performance-chart').getContext('2d');
            performanceChart = new Chart(performanceCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'CPU %',
                        data: [],
                        borderColor: '#ef4444',
                        backgroundColor: 'rgba(239, 68, 68, 0.1)',
                        tension: 0.4
                    }, {
                        label: 'Memory %',
                        data: [],
                        borderColor: '#3b82f6',
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: {
                            labels: {
                                color: '#9ca3af'
                            }
                        }
                    },
                    scales: {
                        x: {
                            ticks: {
                                color: '#9ca3af'
                            },
                            grid: {
                                color: '#374151'
                            }
                        },
                        y: {
                            ticks: {
                                color: '#9ca3af'
                            },
                            grid: {
                                color: '#374151'
                            }
                        }
                    }
                }
            });
            
            // Resources Chart
            const resourcesCtx = document.getElementById('resources-chart').getContext('2d');
            resourcesChart = new Chart(resourcesCtx, {
                type: 'doughnut',
                data: {
                    labels: ['Used', 'Available'],
                    datasets: [{
                        data: [0, 100],
                        backgroundColor: ['#ef4444', '#374151'],
                        borderWidth: 0
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: {
                            labels: {
                                color: '#9ca3af'
                            }
                        }
                    }
                }
            });
        }
        
        function connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/ws/dashboard`);
            
            ws.onopen = function(event) {
                console.log('WebSocket connected');
                document.getElementById('connection-status').innerHTML = 
                    '<span class="status-indicator status-healthy"></span><span>Connected</span>';
                reconnectAttempts = 0;
            };
            
            ws.onmessage = function(event) {
                try {
                    const data = JSON.parse(event.data);
                    handleWebSocketMessage(data);
                } catch (error) {
                    console.error('Error parsing WebSocket message:', error);
                }
            };
            
            ws.onclose = function(event) {
                console.log('WebSocket disconnected');
                document.getElementById('connection-status').innerHTML = 
                    '<span class="status-indicator status-unknown"></span><span>Disconnected</span>';
                
                // Attempt reconnection
                if (reconnectAttempts < maxReconnectAttempts) {
                    setTimeout(() => {
                        reconnectAttempts++;
                        console.log(`Reconnection attempt ${reconnectAttempts}`);
                        connectWebSocket();
                    }, 5000 * reconnectAttempts);
                }
            };
            
            ws.onerror = function(error) {
                console.error('WebSocket error:', error);
            };
        }
        
        function handleWebSocketMessage(data) {
            if (data.type === 'metrics_update') {
                updateDashboard(data.data);
            } else if (data.type === 'health_update') {
                updateHealthStatus(data.data);
            } else if (data.type === 'alert_update') {
                updateAlerts(data.data);
            }
        }
        
        function updateDashboard(data) {
            // Update key metrics
            document.getElementById('uptime-value').textContent = formatUptime(data.uptime_seconds);
            document.getElementById('cpu-value').textContent = data.system_metrics ? 
                `${data.system_metrics.cpu_percent.toFixed(1)}%` : '0%';
            document.getElementById('memory-value').textContent = data.system_metrics ? 
                `${data.system_metrics.memory_used_mb.toFixed(0)}MB` : '0MB';
            document.getElementById('alerts-count').textContent = data.active_alerts.length;
            
            // Update charts
            if (data.system_metrics) {
                updateCharts(data.system_metrics);
            }
            
            // Update performance stats
            updatePerformanceStats(data.performance_stats);
            
            // Update maintenance status
            updateMaintenanceStatus(data.maintenance_status);
        }
        
        function updateCharts(metrics) {
            // Add data point to performance chart
            const now = new Date().toLocaleTimeString();
            
            if (performanceChart.data.labels.length > 20) {
                performanceChart.data.labels.shift();
                performanceChart.data.datasets[0].data.shift();
                performanceChart.data.datasets[1].data.shift();
            }
            
            performanceChart.data.labels.push(now);
            performanceChart.data.datasets[0].data.push(metrics.cpu_percent);
            performanceChart.data.datasets[1].data.push(metrics.memory_percent);
            performanceChart.update('none');
            
            // Update resources chart
            resourcesChart.data.datasets[0].data = [
                metrics.memory_used_mb,
                Math.max(0, 2048 - metrics.memory_used_mb) // Assuming 2GB max
            ];
            resourcesChart.update('none');
        }
        
        function updateHealthStatus(healthData) {
            const container = document.getElementById('health-components');
            container.innerHTML = '';
            
            Object.keys(healthData).forEach(component => {
                const health = healthData[component];
                const statusClass = getStatusClass(health.status);
                
                const element = document.createElement('div');
                element.className = 'flex justify-between items-center py-2 border-b border-gray-700';
                element.innerHTML = `
                    <span>${component}</span>
                    <div class="flex items-center">
                        <span class="status-indicator ${statusClass}"></span>
                        <span class="text-sm">${health.status}</span>
                    </div>
                `;
                container.appendChild(element);
            });
        }
        
        function updateAlerts(alerts) {
            const container = document.getElementById('active-alerts');
            container.innerHTML = '';
            
            if (alerts.length === 0) {
                container.innerHTML = '<p class="text-gray-400">No active alerts</p>';
                return;
            }
            
            alerts.slice(0, 5).forEach(alert => {
                const element = document.createElement('div');
                element.className = `p-3 rounded mb-2 alert-${alert.severity}`;
                element.innerHTML = `
                    <div class="flex justify-between items-start">
                        <div>
                            <div class="font-semibold">${alert.title}</div>
                            <div class="text-sm text-gray-300">${alert.message}</div>
                        </div>
                        <button onclick="resolveAlert('${alert.id}')" class="text-green-400 hover:text-green-300">
                            <i class="fas fa-check"></i>
                        </button>
                    </div>
                `;
                container.appendChild(element);
            });
        }
        
        function updateMaintenanceStatus(maintenanceData) {
            const container = document.getElementById('maintenance-status');
            
            if (maintenanceData.maintenance_mode) {
                container.innerHTML = '<p class="text-yellow-400">Maintenance mode active</p>';
            } else {
                container.innerHTML = '<p class="text-green-400">Normal operations</p>';
            }
            
            if (maintenanceData.next_scheduled_task) {
                const task = maintenanceData.next_scheduled_task;
                container.innerHTML += `
                    <p class="text-sm text-gray-400 mt-2">
                        Next: ${task.name} in ${Math.round(task.hours_remaining)}h
                    </p>
                `;
            }
        }
        
        function updatePerformanceStats(stats) {
            const container = document.getElementById('performance-stats');
            container.innerHTML = `
                <div class="text-sm space-y-1">
                    <div>Total Uptime: ${formatUptime(stats.total_uptime)}</div>
                    <div>Recoveries: ${stats.successful_recoveries}/${stats.recovery_attempts}</div>
                    <div>Maintenance: ${stats.maintenance_cycles} cycles</div>
                    <div>Alerts: ${stats.alerts_sent} sent</div>
                </div>
            `;
        }
        
        function getStatusClass(status) {
            switch(status) {
                case 'healthy': return 'status-healthy';
                case 'degraded': return 'status-degraded';
                case 'critical': return 'status-critical';
                default: return 'status-unknown';
            }
        }
        
        function formatUptime(seconds) {
            const hours = Math.floor(seconds / 3600);
            const days = Math.floor(hours / 24);
            const remainingHours = hours % 24;
            
            if (days > 0) {
                return `${days}d ${remainingHours}h`;
            } else {
                return `${hours}h`;
            }
        }
        
        // API functions
        async function startMaintenance() {
            const reason = prompt('Enter maintenance reason:');
            if (!reason) return;
            
            try {
                const response = await fetch('/api/dashboard/maintenance/start', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({reason})
                });
                
                if (response.ok) {
                    alert('Maintenance mode started');
                } else {
                    alert('Failed to start maintenance mode');
                }
            } catch (error) {
                console.error('Error starting maintenance:', error);
                alert('Error starting maintenance mode');
            }
        }
        
        async function stopMaintenance() {
            const reason = prompt('Enter reason for stopping maintenance:');
            if (!reason) return;
            
            try {
                const response = await fetch('/api/dashboard/maintenance/stop', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({reason})
                });
                
                if (response.ok) {
                    alert('Maintenance mode stopped');
                } else {
                    alert('Failed to stop maintenance mode');
                }
            } catch (error) {
                console.error('Error stopping maintenance:', error);
                alert('Error stopping maintenance mode');
            }
        }
        
        async function createBackup() {
            try {
                const response = await fetch('/api/dashboard/backup', {
                    method: 'POST'
                });
                
                if (response.ok) {
                    const result = await response.json();
                    alert(result.message);
                } else {
                    alert('Failed to create backup');
                }
            } catch (error) {
                console.error('Error creating backup:', error);
                alert('Error creating backup');
            }
        }
        
        async function resolveAlert(alertId) {
            try {
                const response = await fetch(`/api/dashboard/alerts/${alertId}/resolve`, {
                    method: 'POST'
                });
                
                if (response.ok) {
                    // Refresh alerts
                    loadActiveAlerts();
                }
            } catch (error) {
                console.error('Error resolving alert:', error);
            }
        }
        
        async function resolveAllAlerts() {
            if (!confirm('Resolve all active alerts?')) return;
            
            try {
                const response = await fetch('/api/dashboard/alerts/resolve-all', {
                    method: 'POST'
                });
                
                if (response.ok) {
                    loadActiveAlerts();
                }
            } catch (error) {
                console.error('Error resolving all alerts:', error);
            }
        }
        
        function refreshData() {
            loadInitialData();
        }
        
        async function loadInitialData() {
            try {
                const [overview, health, alerts, maintenance, performance] = await Promise.all([
                    fetch('/api/dashboard/overview').then(r => r.json()),
                    fetch('/api/dashboard/health').then(r => r.json()),
                    fetch('/api/dashboard/alerts').then(r => r.json()),
                    fetch('/api/dashboard/maintenance').then(r => r.json()),
                    fetch('/api/dashboard/performance').then(r => r.json())
                ]);
                
                updateDashboard({
                    uptime_seconds: overview.uptime_hours * 3600,
                    system_metrics: performance.latest_metrics,
                    active_alerts: alerts.alerts || [],
                    maintenance_status: maintenance.status,
                    performance_stats: performance.performance_stats,
                    health_status: health.components
                });
            } catch (error) {
                console.error('Error loading initial data:', error);
            }
        }
    </script>
</body>
</html>
"""


def setup_dashboard_static_routes(app: FastAPI):
    """Setup static routes for dashboard assets"""
    
    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard_page():
        """Serve dashboard HTML page"""
        return HTMLResponse(content=DASHBOARD_HTML)
    
    @app.get("/")
    async def root():
        """Redirect root to dashboard"""
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/dashboard")


# ================================
# Integration Functions
# ================================

def integrate_perpetual_dashboard(app: FastAPI, perpetual_manager: PerpetualOperationsManager):
    """Integrate perpetual operations dashboard with FastAPI application"""
    
    # Setup WebSocket manager
    ws_manager = DashboardWebSocketManager()
    
    # Setup routes
    setup_dashboard_routes(app, perpetual_manager)
    setup_websocket_routes(app, perpetual_manager, ws_manager)
    setup_dashboard_static_routes(app)
    
    # Start background tasks
    @app.on_event("startup")
    async def start_dashboard_tasks():
        """Start dashboard background tasks"""
        asyncio.create_task(broadcast_dashboard_updates(perpetual_manager, ws_manager))
    
    return ws_manager