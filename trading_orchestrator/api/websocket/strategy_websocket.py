"""
Strategy WebSocket Manager - Real-time strategy updates

Provides real-time WebSocket connections for:
- Strategy signal updates
- Performance metrics updates
- Alert notifications
- Strategy status changes
"""

import asyncio
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta

from fastapi import WebSocket, WebSocketDisconnect, status, Query
from fastapi.responses import JSONResponse
from loguru import logger

from .manager import WebSocketManager, ConnectionType
from ..utils.json_encoder import JSONEncoder


class StrategyWebSocketManager:
    """
    WebSocket manager specifically for strategy real-time updates
    
    Manages strategy-related real-time communication
    """
    
    def __init__(self):
        self.manager = WebSocketManager()
        self.strategy_signals_cache: Dict[str, List[Dict[str, Any]]] = {}
        self.strategy_performance_cache: Dict[str, Dict[str, Any]] = {}
        self.alert_handlers: Dict[str, List[callable]] = {}
        self._signal_processor_task = None
        self._performance_monitor_task = None
        self._running = False
        
    async def start(self):
        """Start background tasks for real-time updates"""
        if self._running:
            return
        
        self._running = True
        
        # Start background tasks
        self._signal_processor_task = asyncio.create_task(self._process_strategy_signals())
        self._performance_monitor_task = asyncio.create_task(self._monitor_strategy_performance())
        
        logger.info("Strategy WebSocket manager started")
    
    async def stop(self):
        """Stop background tasks"""
        self._running = False
        
        if self._signal_processor_task:
            self._signal_processor_task.cancel()
            
        if self._performance_monitor_task:
            self._performance_monitor_task.cancel()
        
        logger.info("Strategy WebSocket manager stopped")
    
    async def connect_strategy_signals(
        self,
        websocket: WebSocket,
        strategy_id: str,
        user_id: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> str:
        """Connect to strategy signals WebSocket"""
        connection_id = await self.manager.connect(
            websocket=websocket,
            connection_type=ConnectionType.STRATEGY_SIGNALS,
            user_id=user_id,
            strategy_id=strategy_id
        )
        
        # Send recent signals if available
        recent_signals = await self.get_recent_signals(strategy_id, limit=50)
        if recent_signals:
            await self.manager.send_message(
                websocket=websocket,
                message_type="recent_signals",
                data=recent_signals,
                strategy_id=strategy_id
            )
        
        return connection_id
    
    async def connect_strategy_performance(
        self,
        websocket: WebSocket,
        strategy_id: str,
        user_id: str,
        update_frequency: int = 5
    ) -> str:
        """Connect to strategy performance WebSocket"""
        connection_id = await self.manager.connect(
            websocket=websocket,
            connection_type=ConnectionType.STRATEGY_PERFORMANCE,
            user_id=user_id,
            strategy_id=strategy_id
        )
        
        # Send current performance data
        performance_data = await self.get_strategy_performance(strategy_id)
        if performance_data:
            await self.manager.send_message(
                websocket=websocket,
                message_type="performance_update",
                data=performance_data,
                strategy_id=strategy_id
            )
        
        return connection_id
    
    async def connect_strategy_alerts(
        self,
        websocket: WebSocket,
        user_id: str,
        strategy_ids: Optional[List[str]] = None
    ) -> str:
        """Connect to strategy alerts WebSocket"""
        connection_id = await self.manager.connect(
            websocket=websocket,
            connection_type=ConnectionType.STRATEGY_ALERTS,
            user_id=user_id
        )
        
        # Store strategy IDs in metadata for filtering
        if websocket in self.manager.connection_metadata:
            self.manager.connection_metadata[websocket]["alert_strategy_ids"] = strategy_ids
        
        return connection_id
    
    async def connect_system_notifications(
        self,
        websocket: WebSocket,
        user_id: str,
        notification_types: Optional[List[str]] = None
    ) -> str:
        """Connect to system notifications WebSocket"""
        connection_id = await self.manager.connect(
            websocket=websocket,
            connection_type=ConnectionType.SYSTEM_NOTIFICATIONS,
            user_id=user_id
        )
        
        # Store notification types in metadata for filtering
        if websocket in self.manager.connection_metadata:
            self.manager.connection_metadata[websocket]["notification_types"] = notification_types
        
        return connection_id
    
    async def send_strategy_signal(
        self,
        strategy_id: str,
        signal_data: Dict[str, Any]
    ):
        """Send strategy signal to connected clients"""
        # Cache recent signals
        if strategy_id not in self.strategy_signals_cache:
            self.strategy_signals_cache[strategy_id] = []
        
        self.strategy_signals_cache[strategy_id].append({
            **signal_data,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Limit cache size
        if len(self.strategy_signals_cache[strategy_id]) > 1000:
            self.strategy_signals_cache[strategy_id] = self.strategy_signals_cache[strategy_id][-500:]
        
        # Broadcast to strategy signal connections
        await self.manager.send_to_strategy(
            strategy_id=strategy_id,
            connection_type=ConnectionType.STRATEGY_SIGNALS,
            message_type="new_signal",
            data=signal_data
        )
    
    async def send_strategy_performance_update(
        self,
        strategy_id: str,
        performance_data: Dict[str, Any]
    ):
        """Send strategy performance update to connected clients"""
        # Cache performance data
        self.strategy_performance_cache[strategy_id] = {
            **performance_data,
            "last_updated": datetime.utcnow().isoformat()
        }
        
        # Broadcast to strategy performance connections
        await self.manager.send_to_strategy(
            strategy_id=strategy_id,
            connection_type=ConnectionType.STRATEGY_PERFORMANCE,
            message_type="performance_update",
            data=performance_data
        )
    
    async def send_strategy_alert(
        self,
        strategy_id: str,
        alert_data: Dict[str, Any],
        user_id: Optional[str] = None
    ):
        """Send strategy alert to connected clients"""
        alert_message = {
            "strategy_id": strategy_id,
            "alert_type": alert_data.get("type"),
            "severity": alert_data.get("severity", "info"),
            "message": alert_data.get("message"),
            "data": alert_data.get("data"),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Broadcast to all alert connections
        if user_id:
            await self.manager.send_to_user(
                user_id=user_id,
                connection_type=ConnectionType.STRATEGY_ALERTS,
                message_type="strategy_alert",
                data=alert_message,
                strategy_id=strategy_id
            )
        else:
            await self.manager.broadcast_message(
                connection_type=ConnectionType.STRATEGY_ALERTS,
                message_type="strategy_alert",
                data=alert_message,
                strategy_id=strategy_id
            )
    
    async def send_system_notification(
        self,
        notification_data: Dict[str, Any],
        user_id: Optional[str] = None,
        notification_type: Optional[str] = None
    ):
        """Send system notification to connected clients"""
        notification_message = {
            "type": notification_data.get("type", "system"),
            "severity": notification_data.get("severity", "info"),
            "title": notification_data.get("title"),
            "message": notification_data.get("message"),
            "data": notification_data.get("data"),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if user_id:
            await self.manager.send_to_user(
                user_id=user_id,
                connection_type=ConnectionType.SYSTEM_NOTIFICATIONS,
                message_type="system_notification",
                data=notification_message
            )
        else:
            await self.manager.broadcast_message(
                connection_type=ConnectionType.SYSTEM_NOTIFICATIONS,
                message_type="system_notification",
                data=notification_message
            )
    
    async def get_recent_signals(
        self,
        strategy_id: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get recent signals for strategy"""
        signals = self.strategy_signals_cache.get(strategy_id, [])
        return signals[-limit:] if signals else []
    
    async def get_strategy_performance(
        self,
        strategy_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get current performance data for strategy"""
        return self.strategy_performance_cache.get(strategy_id)
    
    def get_connection_count(self) -> int:
        """Get total connection count"""
        return self.manager.get_connection_count()
    
    def get_connection_count_by_type(self, connection_type: str) -> int:
        """Get connection count by type"""
        return self.manager.get_connection_count_by_type(connection_type)
    
    async def disconnect(self, websocket: WebSocket):
        """Disconnect WebSocket"""
        await self.manager.disconnect(websocket)
    
    async def disconnect_all(self):
        """Disconnect all WebSockets"""
        await self.manager.disconnect_all()
    
    async def _process_strategy_signals(self):
        """Background task to process strategy signals"""
        while self._running:
            try:
                # This would integrate with your strategy signal processing
                # For now, just sleep
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in signal processor: {e}")
                await asyncio.sleep(5)
    
    async def _monitor_strategy_performance(self):
        """Background task to monitor strategy performance"""
        while self._running:
            try:
                # This would integrate with your strategy performance monitoring
                # For now, just sleep
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"Error in performance monitor: {e}")
                await asyncio.sleep(10)


# Global instance
_strategy_ws_manager = None

async def get_strategy_websocket_manager() -> StrategyWebSocketManager:
    """Get strategy WebSocket manager instance"""
    global _strategy_ws_manager
    if _strategy_ws_manager is None:
        _strategy_ws_manager = StrategyWebSocketManager()
        await _strategy_ws_manager.start()
    return _strategy_ws_manager
