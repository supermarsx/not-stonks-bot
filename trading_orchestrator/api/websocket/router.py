"""
WebSocket Router - Real-time endpoints for strategy updates

Provides WebSocket endpoints for:
- Strategy signal streaming
- Performance metrics updates
- Alert notifications
- System notifications
"""

from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, status
from fastapi.responses import JSONResponse
from loguru import logger

from ..websocket.strategy_websocket import get_strategy_websocket_manager
from ..routers.dependencies import get_current_user


router = APIRouter(prefix="/ws", tags=["websocket"])


@router.websocket("/strategies/{strategy_id}/signals")
async def websocket_strategy_signals(
    websocket: WebSocket,
    strategy_id: str,
    user_id: Optional[str] = Query(None, description="User ID for authentication"),
    token: Optional[str] = Query(None, description="Authentication token")
):
    """WebSocket endpoint for real-time strategy signals"""
    
    # Authenticate user (implement your authentication logic here)
    current_user = None
    try:
        # This would validate the token and user_id
        # For now, create a demo user
        current_user = {
            "id": user_id or "demo_user",
            "name": "Demo User",
            "role": "trader"
        }
    except Exception as e:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Authentication failed")
        return
    
    # Get strategy WebSocket manager
    strategy_ws_manager = await get_strategy_websocket_manager()
    
    try:
        # Connect to strategy signals
        connection_id = await strategy_ws_manager.connect_strategy_signals(
            websocket=websocket,
            strategy_id=strategy_id,
            user_id=current_user["id"],
            filters={}  # Add any filters you need
        )
        
        logger.info(f"Strategy signals WebSocket connected: {connection_id}")
        
        # Keep connection alive
        while True:
            try:
                # Receive messages from client
                message = await websocket.receive_text()
                
                # Process client messages (ping, subscribe, unsubscribe, etc.)
                await _handle_client_message(websocket, message, strategy_ws_manager, strategy_id)
                
            except WebSocketDisconnect:
                logger.info(f"Strategy signals WebSocket disconnected: {connection_id}")
                break
            except Exception as e:
                logger.error(f"Error in strategy signals WebSocket: {e}")
                await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
                break
                
    except WebSocketDisconnect:
        logger.info(f"Strategy signals WebSocket disconnected: {connection_id}")
    except Exception as e:
        logger.error(f"Strategy signals WebSocket error: {e}")
        await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
    finally:
        await strategy_ws_manager.disconnect(websocket)


@router.websocket("/strategies/{strategy_id}/performance")
async def websocket_strategy_performance(
    websocket: WebSocket,
    strategy_id: str,
    user_id: Optional[str] = Query(None, description="User ID for authentication"),
    token: Optional[str] = Query(None, description="Authentication token"),
    frequency: int = Query(5, ge=1, le=60, description="Update frequency in seconds")
):
    """WebSocket endpoint for real-time strategy performance metrics"""
    
    # Authenticate user
    current_user = None
    try:
        # This would validate the token and user_id
        current_user = {
            "id": user_id or "demo_user",
            "name": "Demo User",
            "role": "trader"
        }
    except Exception as e:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Authentication failed")
        return
    
    # Get strategy WebSocket manager
    strategy_ws_manager = await get_strategy_websocket_manager()
    
    try:
        # Connect to strategy performance
        connection_id = await strategy_ws_manager.connect_strategy_performance(
            websocket=websocket,
            strategy_id=strategy_id,
            user_id=current_user["id"],
            update_frequency=frequency
        )
        
        logger.info(f"Strategy performance WebSocket connected: {connection_id}")
        
        # Send periodic performance updates
        while True:
            try:
                # Receive messages from client
                message = await websocket.receive_text()
                
                # Process client messages
                await _handle_client_message(websocket, message, strategy_ws_manager, strategy_id)
                
                # Send performance update
                # This would integrate with your actual performance monitoring
                performance_data = {
                    "strategy_id": strategy_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "performance": {
                        "total_return": 0.0,  # Would be actual performance data
                        "sharpe_ratio": 0.0,
                        "max_drawdown": 0.0,
                        "win_rate": 0.0,
                        "total_trades": 0
                    },
                    "current_status": "running"
                }
                
                await strategy_ws_manager.manager.send_message(
                    websocket=websocket,
                    message_type="performance_update",
                    data=performance_data,
                    strategy_id=strategy_id
                )
                
                # Wait for next update
                await asyncio.sleep(frequency)
                
            except WebSocketDisconnect:
                logger.info(f"Strategy performance WebSocket disconnected: {connection_id}")
                break
            except Exception as e:
                logger.error(f"Error in strategy performance WebSocket: {e}")
                await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
                break
                
    except WebSocketDisconnect:
        logger.info(f"Strategy performance WebSocket disconnected: {connection_id}")
    except Exception as e:
        logger.error(f"Strategy performance WebSocket error: {e}")
        await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
    finally:
        await strategy_ws_manager.disconnect(websocket)


@router.websocket("/strategies/alerts")
async def websocket_strategy_alerts(
    websocket: WebSocket,
    user_id: Optional[str] = Query(None, description="User ID for authentication"),
    token: Optional[str] = Query(None, description="Authentication token"),
    strategy_ids: Optional[str] = Query(None, description="Comma-separated strategy IDs")
):
    """WebSocket endpoint for strategy alert notifications"""
    
    # Authenticate user
    current_user = None
    try:
        current_user = {
            "id": user_id or "demo_user",
            "name": "Demo User",
            "role": "trader"
        }
    except Exception as e:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Authentication failed")
        return
    
    # Parse strategy IDs
    strategy_id_list = []
    if strategy_ids:
        strategy_id_list = [sid.strip() for sid in strategy_ids.split(",")]
    
    # Get strategy WebSocket manager
    strategy_ws_manager = await get_strategy_websocket_manager()
    
    try:
        # Connect to strategy alerts
        connection_id = await strategy_ws_manager.connect_strategy_alerts(
            websocket=websocket,
            user_id=current_user["id"],
            strategy_ids=strategy_id_list
        )
        
        logger.info(f"Strategy alerts WebSocket connected: {connection_id}")
        
        # Keep connection alive and handle messages
        while True:
            try:
                # Receive messages from client
                message = await websocket.receive_text()
                
                # Process client messages
                await _handle_client_message(websocket, message, strategy_ws_manager)
                
            except WebSocketDisconnect:
                logger.info(f"Strategy alerts WebSocket disconnected: {connection_id}")
                break
            except Exception as e:
                logger.error(f"Error in strategy alerts WebSocket: {e}")
                await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
                break
                
    except WebSocketDisconnect:
        logger.info(f"Strategy alerts WebSocket disconnected: {connection_id}")
    except Exception as e:
        logger.error(f"Strategy alerts WebSocket error: {e}")
        await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
    finally:
        await strategy_ws_manager.disconnect(websocket)


@router.websocket("/system/notifications")
async def websocket_system_notifications(
    websocket: WebSocket,
    user_id: Optional[str] = Query(None, description="User ID for authentication"),
    token: Optional[str] = Query(None, description="Authentication token"),
    types: Optional[str] = Query(None, description="Comma-separated notification types")
):
    """WebSocket endpoint for system notifications"""
    
    # Authenticate user
    current_user = None
    try:
        current_user = {
            "id": user_id or "demo_user",
            "name": "Demo User",
            "role": "trader"
        }
    except Exception as e:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Authentication failed")
        return
    
    # Parse notification types
    notification_type_list = []
    if types:
        notification_type_list = [t.strip() for t in types.split(",")]
    
    # Get strategy WebSocket manager
    strategy_ws_manager = await get_strategy_websocket_manager()
    
    try:
        # Connect to system notifications
        connection_id = await strategy_ws_manager.connect_system_notifications(
            websocket=websocket,
            user_id=current_user["id"],
            notification_types=notification_type_list
        )
        
        logger.info(f"System notifications WebSocket connected: {connection_id}")
        
        # Keep connection alive and handle messages
        while True:
            try:
                # Receive messages from client
                message = await websocket.receive_text()
                
                # Process client messages
                await _handle_client_message(websocket, message, strategy_ws_manager)
                
            except WebSocketDisconnect:
                logger.info(f"System notifications WebSocket disconnected: {connection_id}")
                break
            except Exception as e:
                logger.error(f"Error in system notifications WebSocket: {e}")
                await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
                break
                
    except WebSocketDisconnect:
        logger.info(f"System notifications WebSocket disconnected: {connection_id}")
    except Exception as e:
        logger.error(f"System notifications WebSocket error: {e}")
        await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
    finally:
        await strategy_ws_manager.disconnect(websocket)


async def _handle_client_message(
    websocket: WebSocket,
    message: str,
    strategy_ws_manager: 'StrategyWebSocketManager',
    strategy_id: Optional[str] = None
):
    """Handle incoming client messages"""
    try:
        import json
        
        # Parse message
        data = json.loads(message)
        message_type = data.get("type")
        
        if message_type == "ping":
            # Respond to ping
            await strategy_ws_manager.manager.send_message(
                websocket=websocket,
                message_type="pong",
                data={"timestamp": datetime.utcnow().isoformat()},
                strategy_id=strategy_id
            )
            
        elif message_type == "subscribe":
            # Handle subscription updates
            subscriptions = data.get("subscriptions", [])
            # Process subscriptions
            pass
            
        elif message_type == "unsubscribe":
            # Handle unsubscription
            subscriptions = data.get("subscriptions", [])
            # Process unsubscriptions
            pass
            
        elif message_type == "request_performance":
            # Send performance data on request
            if strategy_id:
                performance_data = await strategy_ws_manager.get_strategy_performance(strategy_id)
                if performance_data:
                    await strategy_ws_manager.manager.send_message(
                        websocket=websocket,
                        message_type="performance_data",
                        data=performance_data,
                        strategy_id=strategy_id
                    )
        
    except json.JSONDecodeError:
        logger.warning(f"Invalid JSON message received: {message}")
    except Exception as e:
        logger.error(f"Error handling client message: {e}")
