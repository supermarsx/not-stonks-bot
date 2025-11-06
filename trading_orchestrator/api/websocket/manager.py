"""
WebSocket Manager for real-time API communication

Provides connection management for WebSocket endpoints:
- Connection tracking and management
- Message broadcasting
- Authentication for WebSocket connections
"""

import json
import asyncio
from typing import Dict, List, Set, Optional, Any, Callable
from datetime import datetime
from enum import Enum

from fastapi import WebSocket, WebSocketDisconnect, status
from loguru import logger

from ..schemas.strategies import WebSocketMessage
from ..utils.json_encoder import JSONEncoder


class ConnectionType(str, Enum):
    """WebSocket connection types"""
    STRATEGY_SIGNALS = "strategy_signals"
    STRATEGY_PERFORMANCE = "strategy_performance"
    STRATEGY_ALERTS = "strategy_alerts"
    SYSTEM_NOTIFICATIONS = "system_notifications"
    TRADE_EXECUTION = "trade_execution"


class WebSocketManager:
    """
    WebSocket connection manager for real-time communication
    
    Manages WebSocket connections and message broadcasting
    """
    
    def __init__(self):
        self.connections: Dict[str, Set[WebSocket]] = {}
        self.connection_metadata: Dict[WebSocket, Dict[str, Any]] = {}
        self.message_handlers: Dict[str, Callable] = {}
        self._lock = asyncio.Lock()
        
    async def connect(
        self,
        websocket: WebSocket,
        connection_type: str,
        user_id: Optional[str] = None,
        strategy_id: Optional[str] = None
    ) -> str:
        """Accept WebSocket connection and register it"""
        await websocket.accept()
        
        connection_id = f"{connection_type}:{websocket.client.host}:{websocket.client.port}:{datetime.utcnow().timestamp()}"
        
        async with self._lock:
            # Add connection to tracking
            if connection_type not in self.connections:
                self.connections[connection_type] = set()
            
            self.connections[connection_type].add(websocket)
            
            # Store metadata
            self.connection_metadata[websocket] = {
                "id": connection_id,
                "type": connection_type,
                "user_id": user_id,
                "strategy_id": strategy_id,
                "connected_at": datetime.utcnow(),
                "last_message": datetime.utcnow(),
                "message_count": 0
            }
            
            logger.info(f"WebSocket connected: {connection_id} ({connection_type})")
        
        return connection_id
    
    async def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection from tracking"""
        async with self._lock:
            if websocket in self.connection_metadata:
                metadata = self.connection_metadata[websocket]
                connection_type = metadata["type"]
                
                # Remove from connections
                if connection_type in self.connections:
                    self.connections[connection_type].discard(websocket)
                    
                    # Clean up empty connection types
                    if not self.connections[connection_type]:
                        del self.connections[connection_type]
                
                # Log disconnection
                logger.info(f"WebSocket disconnected: {metadata['id']} ({connection_type})")
                
                # Remove metadata
                del self.connection_metadata[websocket]
    
    async def disconnect_all(self):
        """Disconnect all WebSocket connections"""
        async with self._lock:
            for connection_type, connections in self.connections.items():
                for websocket in connections.copy():
                    try:
                        await websocket.close()
                    except Exception:
                        pass
            
            self.connections.clear()
            self.connection_metadata.clear()
            logger.info("All WebSocket connections closed")
    
    def get_connection_count(self) -> int:
        """Get total number of active connections"""
        return sum(len(connections) for connections in self.connections.values())
    
    def get_connection_count_by_type(self, connection_type: str) -> int:
        """Get number of connections for specific type"""
        return len(self.connections.get(connection_type, set()))
    
    async def send_message(
        self,
        websocket: WebSocket,
        message_type: str,
        data: Any,
        strategy_id: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ):
        """Send message to specific WebSocket connection"""
        try:
            message = WebSocketMessage(
                type=message_type,
                strategy_id=strategy_id,
                data=data,
                timestamp=timestamp or datetime.utcnow()
            )
            
            await websocket.send_text(message.json(by_alias=True))
            
            # Update metadata
            if websocket in self.connection_metadata:
                self.connection_metadata[websocket]["last_message"] = datetime.utcnow()
                self.connection_metadata[websocket]["message_count"] += 1
                
        except Exception as e:
            logger.error(f"Error sending WebSocket message: {e}")
            # Remove failed connection
            await self.disconnect(websocket)
    
    async def broadcast_message(
        self,
        connection_type: str,
        message_type: str,
        data: Any,
        strategy_id: Optional[str] = None,
        exclude_websocket: Optional[WebSocket] = None
    ):
        """Broadcast message to all connections of specific type"""
        if connection_type not in self.connections:
            return
        
        failed_connections = []
        
        async with self._lock:
            connections = self.connections[connection_type].copy()
        
        # Send to all connections
        for websocket in connections:
            if exclude_websocket and websocket == exclude_websocket:
                continue
                
            try:
                await self.send_message(
                    websocket=websocket,
                    message_type=message_type,
                    data=data,
                    strategy_id=strategy_id
                )
            except Exception as e:
                logger.error(f"Failed to send message to WebSocket: {e}")
                failed_connections.append(websocket)
        
        # Remove failed connections
        for websocket in failed_connections:
            await self.disconnect(websocket)
    
    async def send_to_user(
        self,
        user_id: str,
        connection_type: str,
        message_type: str,
        data: Any,
        strategy_id: Optional[str] = None
    ):
        """Send message to all connections for specific user"""
        user_connections = []
        
        async with self._lock:
            for websocket, metadata in self.connection_metadata.items():
                if (metadata.get("user_id") == user_id and 
                    metadata.get("type") == connection_type):
                    user_connections.append(websocket)
        
        for websocket in user_connections:
            try:
                await self.send_message(
                    websocket=websocket,
                    message_type=message_type,
                    data=data,
                    strategy_id=strategy_id
                )
            except Exception as e:
                logger.error(f"Failed to send message to user WebSocket: {e}")
                await self.disconnect(websocket)
    
    async def send_to_strategy(
        self,
        strategy_id: str,
        connection_type: str,
        message_type: str,
        data: Any
    ):
        """Send message to all connections for specific strategy"""
        strategy_connections = []
        
        async with self._lock:
            for websocket, metadata in self.connection_metadata.items():
                if (metadata.get("strategy_id") == strategy_id and 
                    metadata.get("type") == connection_type):
                    strategy_connections.append(websocket)
        
        for websocket in strategy_connections:
            try:
                await self.send_message(
                    websocket=websocket,
                    message_type=message_type,
                    data=data,
                    strategy_id=strategy_id
                )
            except Exception as e:
                logger.error(f"Failed to send message to strategy WebSocket: {e}")
                await self.disconnect(websocket)
    
    def get_connection_info(self, websocket: WebSocket) -> Optional[Dict[str, Any]]:
        """Get connection metadata"""
        return self.connection_metadata.get(websocket)
    
    async def get_user_connections(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all connections for specific user"""
        async with self._lock:
            user_connections = []
            for websocket, metadata in self.connection_metadata.items():
                if metadata.get("user_id") == user_id:
                    user_connections.append({
                        "id": metadata["id"],
                        "type": metadata["type"],
                        "strategy_id": metadata.get("strategy_id"),
                        "connected_at": metadata["connected_at"],
                        "message_count": metadata["message_count"],
                        "last_message": metadata["last_message"]
                    })
            return user_connections
    
    async def handle_message(
        self,
        websocket: WebSocket,
        message_data: Dict[str, Any]
    ) -> bool:
        """Handle incoming message from WebSocket"""
        try:
            message_type = message_data.get("type")
            
            if message_type in self.message_handlers:
                await self.message_handlers[message_type](websocket, message_data)
                return True
            
            logger.warning(f"No handler for message type: {message_type}")
            return False
            
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")
            return False
    
    def register_message_handler(self, message_type: str, handler: Callable):
        """Register handler for specific message type"""
        self.message_handlers[message_type] = handler
    
    async def ping_connections(self):
        """Send ping to all active connections"""
        async with self._lock:
            connections = list(self.connections.values())
        
        for connection_group in connections:
            for websocket in connection_group:
                try:
                    await websocket.send_text(JSONEncoder.encode({
                        "type": "ping",
                        "timestamp": datetime.utcnow().isoformat()
                    }))
                except Exception as e:
                    logger.error(f"Failed to ping WebSocket: {e}")
                    await self.disconnect(websocket)