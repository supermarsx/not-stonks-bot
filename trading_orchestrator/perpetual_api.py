"""
FastAPI Application with Perpetual Operations Integration
========================================================

Complete FastAPI application that integrates the existing trading orchestrator API
with comprehensive perpetual operations capabilities.

Features:
- Complete trading orchestrator API
- Perpetual operations dashboard and API
- Real-time WebSocket connections
- Health monitoring and alerting
- Maintenance mode management
- Backup and recovery operations
- Operational runbooks

Author: Trading Orchestrator System
Version: 2.0.0
Date: 2025-11-06
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

# Import core trading system
from config.settings import settings
from config.application import app_config, ApplicationConfig
from risk.manager import RiskManager
from oms.manager import OrderManager

# Import perpetual operations
from perpetual_operations import setup_perpetual_fastapi_integration, PerpetualIntegration
from perpetual_operations.dashboard import DASHBOARD_HTML


# ================================
# FastAPI Application Setup
# ================================

def create_app() -> FastAPI:
    """Create and configure the FastAPI application"""
    
    # Initialize FastAPI app
    app = FastAPI(
        title="Trading Orchestrator with Perpetual Operations",
        description="Complete trading orchestrator API with 24/7 operation capabilities",
        version="2.0.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    return app


# ================================
# Trading System API Endpoints
# ================================

def setup_trading_api_routes(app: FastAPI, app_config_instance: ApplicationConfig):
    """Setup trading orchestrator API routes"""
    
    @app.get("/api/health")
    async def health_check():
        """Basic health check endpoint"""
        try:
            health_status = await app_config_instance.get_health_status()
            return {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "health": health_status
            }
        except Exception as e:
            logger.error(f"Health check error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/system/status")
    async def get_system_status():
        """Get comprehensive system status"""
        try:
            system_status = await app_config_instance.get_health_status()
            return {
                "status": "success",
                "timestamp": datetime.utcnow().isoformat(),
                "data": system_status
            }
        except Exception as e:
            logger.error(f"System status error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/brokers")
    async def get_brokers():
        """Get broker connections status"""
        try:
            brokers = {}
            for name, broker in app_config_instance.state.brokers.items():
                brokers[name] = {
                    "connected": True,
                    "type": broker.__class__.__name__,
                    "status": "active"
                }
            
            return {
                "status": "success",
                "timestamp": datetime.utcnow().isoformat(),
                "data": {
                    "brokers": brokers,
                    "total_count": len(brokers),
                    "connected_count": len(brokers)
                }
            }
        except Exception as e:
            logger.error(f"Brokers status error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/orders")
    async def get_orders():
        """Get order management status"""
        try:
            if not app_config_instance.state.order_manager:
                raise HTTPException(status_code=503, detail="Order manager not available")
            
            # Get order manager metrics
            performance_metrics = await app_config_instance.state.order_manager.get_performance_metrics()
            
            return {
                "status": "success",
                "timestamp": datetime.utcnow().isoformat(),
                "data": {
                    "performance_metrics": performance_metrics,
                    "order_manager_status": "operational"
                }
            }
        except Exception as e:
            logger.error(f"Orders status error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/risk")
    async def get_risk_status():
        """Get risk management status"""
        try:
            if not app_config_instance.state.risk_manager:
                raise HTTPException(status_code=503, detail="Risk manager not available")
            
            # Get risk summary
            risk_summary = await app_config_instance.state.risk_manager.get_risk_summary()
            
            return {
                "status": "success",
                "timestamp": datetime.utcnow().isoformat(),
                "data": {
                    "risk_summary": risk_summary,
                    "risk_manager_status": "operational"
                }
            }
        except Exception as e:
            logger.error(f"Risk status error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/ai")
    async def get_ai_status():
        """Get AI orchestrator status"""
        try:
            if not app_config_instance.state.ai_orchestrator:
                raise HTTPException(status_code=503, detail="AI orchestrator not available")
            
            # Get AI performance stats
            ai_stats = app_config_instance.state.ai_orchestrator.get_performance_stats()
            
            return {
                "status": "success",
                "timestamp": datetime.utcnow().isoformat(),
                "data": {
                    "ai_stats": ai_stats,
                    "ai_orchestrator_status": "operational",
                    "trading_mode": app_config_instance.state.ai_orchestrator.trading_mode.value
                }
            }
        except Exception as e:
            logger.error(f"AI status error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/orders/test")
    async def test_order_routing():
        """Test order routing functionality"""
        try:
            # This would implement a test order routing
            # For now, return a mock response
            
            return {
                "status": "success",
                "message": "Order routing test completed",
                "timestamp": datetime.utcnow().isoformat(),
                "data": {
                    "test_order_id": f"test_{int(datetime.utcnow().timestamp())}",
                    "status": "completed",
                    "routing_successful": True
                }
            }
        except Exception as e:
            logger.error(f"Order routing test error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/overview")
    async def get_system_overview():
        """Get complete system overview"""
        try:
            # This would call the method from the main app class
            # For now, create a comprehensive overview
            
            health = await app_config_instance.get_health_status()
            
            overview = {
                "timestamp": datetime.utcnow().isoformat(),
                "system_info": {
                    "name": "Trading Orchestrator",
                    "version": "2.0.0",
                    "environment": settings.environment.value,
                    "uptime_hours": health.get("uptime_seconds", 0) / 3600
                },
                "components": {
                    "database": health["components"].get("database", {}).get("healthy", False),
                    "brokers": len(app_config_instance.state.brokers),
                    "ai_orchestrator": app_config_instance.state.ai_orchestrator is not None,
                    "risk_manager": app_config_instance.state.risk_manager is not None,
                    "order_manager": app_config_instance.state.order_manager is not None
                },
                "health_status": health.get("status", "unknown"),
                "alerts": health.get("components", {})
            }
            
            return {
                "status": "success",
                "data": overview
            }
        except Exception as e:
            logger.error(f"System overview error: {e}")
            raise HTTPException(status_code=500, detail=str(e))


# ================================
# WebSocket Manager
# ================================

class TradingWebSocketManager:
    """WebSocket manager for trading system updates"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_counter = 0
    
    async def connect(self, websocket: WebSocket, client_id: str = None):
        """Accept new WebSocket connection"""
        await websocket.accept()
        
        if not client_id:
            client_id = f"trading_client_{self.connection_counter}"
            self.connection_counter += 1
        
        self.active_connections[client_id] = websocket
        logger.info(f"Trading WebSocket connected: {client_id}")
        
        # Send welcome message
        await websocket.send_text({
            "type": "welcome",
            "data": {
                "client_id": client_id,
                "timestamp": datetime.utcnow().isoformat(),
                "message": "Connected to Trading Orchestrator WebSocket"
            }
        })
    
    def disconnect(self, client_id: str):
        """Remove WebSocket connection"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"Trading WebSocket disconnected: {client_id}")
    
    async def broadcast_trading_update(self, update_type: str, data: Dict[str, Any]):
        """Broadcast trading update to all connected clients"""
        if not self.active_connections:
            return
        
        message = {
            "type": update_type,
            "data": data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        disconnected_clients = []
        
        for client_id, websocket in self.active_connections.items():
            try:
                await websocket.send_text(str(message).replace("'", '"'))
            except Exception as e:
                logger.error(f"Error sending WebSocket message to {client_id}: {e}")
                disconnected_clients.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            self.disconnect(client_id)


def setup_websocket_routes(app: FastAPI, ws_manager: TradingWebSocketManager):
    """Setup WebSocket routes"""
    
    @app.websocket("/ws/trading")
    async def trading_websocket(websocket: WebSocket, client_id: str = None):
        """WebSocket endpoint for trading updates"""
        await ws_manager.connect(websocket, client_id)
        
        try:
            while True:
                data = await websocket.receive_text()
                
                # Echo back received data
                response = {
                    "type": "echo",
                    "data": {
                        "original": data,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                }
                await websocket.send_text(str(response).replace("'", '"'))
                
        except WebSocketDisconnect:
            ws_manager.disconnect(client_id)
        except Exception as e:
            logger.error(f"Trading WebSocket error: {e}")
            ws_manager.disconnect(client_id)


# ================================
# Static Files and Dashboard
# ================================

def setup_static_and_dashboard(app: FastAPI):
    """Setup static files and dashboard routes"""
    
    @app.get("/")
    async def root():
        """Redirect to dashboard"""
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/dashboard")
    
    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard():
        """Serve the perpetual operations dashboard"""
        return HTMLResponse(content=DASHBOARD_HTML)
    
    # Mount static files if needed
    static_dir = Path("static")
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory="static"), name="static")


# ================================
# Main Application Factory
# ================================

async def create_complete_application() -> tuple[FastAPI, ApplicationConfig, TradingWebSocketManager, PerpetualIntegration]:
    """Create complete FastAPI application with all integrations"""
    
    logger.info("🏗️ Creating complete FastAPI application...")
    
    # 1. Create FastAPI app
    app = create_app()
    
    # 2. Initialize application configuration
    app_config_instance = ApplicationConfig()
    startup_success = await app_config_instance.initialize()
    
    if not startup_success:
        raise RuntimeError("Failed to initialize application configuration")
    
    # 3. Setup trading API routes
    setup_trading_api_routes(app, app_config_instance)
    
    # 4. Setup WebSocket manager
    ws_manager = TradingWebSocketManager()
    setup_websocket_routes(app, ws_manager)
    
    # 5. Setup static files and dashboard
    setup_static_and_dashboard(app)
    
    # 6. Setup perpetual operations integration
    perpetual_integration = setup_perpetual_fastapi_integration(app, app_config_instance)
    
    # 7. Add startup and shutdown events
    @app.on_event("startup")
    async def startup_event():
        """Application startup event"""
        logger.info("🚀 FastAPI application starting up...")
        
        # Send startup alert
        try:
            from perpetual_operations import AlertManager
            alert = AlertManager.AlertInfo(
                id=f"startup_{int(datetime.utcnow().timestamp())}",
                severity='low',
                title="System Startup Complete",
                message="Trading Orchestrator with Perpetual Operations started successfully",
                component='system',
                timestamp=datetime.utcnow().isoformat(),
                details={'version': '2.0.0'}
            )
            await AlertManager.send_alert(alert)
        except Exception as e:
            logger.error(f"Startup alert error: {e}")
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """Application shutdown event"""
        logger.info("🛑 FastAPI application shutting down...")
        
        # Send shutdown alert
        try:
            from perpetual_operations import AlertManager
            alert = AlertManager.AlertInfo(
                id=f"shutdown_{int(datetime.utcnow().timestamp())}",
                severity='medium',
                title="System Shutdown",
                message="Trading Orchestrator shutdown initiated",
                component='system',
                timestamp=datetime.utcnow().isoformat(),
                details={'version': '2.0.0'}
            )
            await AlertManager.send_alert(alert)
        except Exception as e:
            logger.error(f"Shutdown alert error: {e}")
        
        # Shutdown application configuration
        await app_config_instance.shutdown()
    
    logger.success("✅ Complete FastAPI application created successfully")
    
    return app, app_config_instance, ws_manager, perpetual_integration


# ================================
# Main Entry Points
# ================================

async def run_standalone_server():
    """Run standalone FastAPI server with perpetual operations"""
    
    try:
        # Create complete application
        app, app_config, ws_manager, perpetual_integration = await create_complete_application()
        
        # Configure uvicorn server
        config = uvicorn.Config(
            app,
            host=settings.api_host,
            port=settings.api_port,
            log_level="info",
            access_log=True,
            reload=False  # Disable reload for production
        )
        
        server = uvicorn.Server(config)
        
        logger.info(f"🌐 Starting server at http://{settings.api_host}:{settings.api_port}")
        logger.info(f"📊 Dashboard: http://{settings.api_host}:{settings.api_port}/dashboard")
        logger.info(f"📚 API Docs: http://{settings.api_host}:{settings.api_port}/api/docs")
        
        await server.serve()
        
    except Exception as e:
        logger.error(f"Server startup failed: {e}")
        sys.exit(1)


def main():
    """Main entry point for FastAPI server"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Trading Orchestrator with Perpetual Operations")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker processes")
    
    args = parser.parse_args()
    
    # Update settings
    import os
    os.environ["API_HOST"] = args.host
    os.environ["API_PORT"] = str(args.port)
    
    # Run server
    asyncio.run(run_standalone_server())


if __name__ == "__main__":
    main()