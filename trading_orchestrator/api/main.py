"""
Main FastAPI Application for Trading Orchestrator Strategy API

Provides REST API endpoints and WebSocket connections for:
- Strategy management and configuration
- Real-time strategy monitoring
- Backtesting and performance analysis
- Strategy comparison and selection
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException, Depends, status, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from loguru import logger

# API Components
from .auth.authentication import AuthenticationManager
from .auth.authorization import AuthorizationManager
from .database.models import StrategyDB, StrategyPerformanceDB
from .websocket.manager import WebSocketManager
from .websocket.strategy_websocket import StrategyWebSocketManager
from .utils.response_formatter import ResponseFormatter
from .utils.error_handlers import setup_error_handlers

# Strategy Components  
from ..strategies.strategy_management import StrategySelector
from ..strategies.enhanced_backtesting import EnhancedBacktestEngine
from ..strategies.base import StrategyStatus

# Configuration
from ..config.settings import settings


# Global instances
authentication_manager = None
authorization_manager = None
websocket_manager = None
strategy_websocket_manager = None
strategy_selector = None
backtest_engine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    
    # Startup
    logger.info("🚀 Starting Trading Strategy API...")
    
    global authentication_manager, authorization_manager, websocket_manager, strategy_websocket_manager, strategy_selector, backtest_engine
    
    try:
        # Initialize authentication and authorization
        authentication_manager = AuthenticationManager()
        authorization_manager = AuthorizationManager()
        
        # Initialize WebSocket managers
        websocket_manager = WebSocketManager()
        strategy_websocket_manager = StrategyWebSocketManager()
        
        # Initialize strategy components
        strategy_selector = StrategySelector()
        backtest_engine = EnhancedBacktestEngine()
        
        # Setup error handlers
        setup_error_handlers(app)
        
        logger.success("✅ API startup completed")
        
    except Exception as e:
        logger.exception(f"❌ API startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🔄 Shutting down Trading Strategy API...")
    
    try:
        # Close WebSocket connections
        await websocket_manager.disconnect_all()
        await strategy_websocket_manager.disconnect_all()
        
        logger.success("✅ API shutdown completed")
        
    except Exception as e:
        logger.error(f"❌ API shutdown error: {e}")


# Create FastAPI application
app = FastAPI(
    title="Trading Orchestrator Strategy API",
    description="""
    Comprehensive API for trading strategy management and analysis.
    
    Features:
    - Complete CRUD operations for trading strategies
    - Real-time WebSocket connections for live updates
    - Advanced backtesting with Monte Carlo simulation
    - Strategy comparison and ranking
    - Performance analysis and metrics
    - Authentication and authorization
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api_cors_origins.split(",") if settings.api_cors_origins else ["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Trusted host middleware (production security)
if settings.environment == "production":
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.api_allowed_hosts.split(",") if settings.api_allowed_hosts else ["localhost"]
    )

# Include API routers
from .routers.strategies import router as strategies_router
from .websocket.router import router as websocket_router

app.include_router(strategies_router)
app.include_router(websocket_router)

# Security
security = HTTPBearer(auto_error=False)

# Global dependencies
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Optional[Dict[str, Any]]:
    """Get current authenticated user"""
    if not credentials:
        return None
    
    try:
        user = await authentication_manager.authenticate_token(credentials.credentials)
        return user
    except Exception as e:
        logger.warning(f"Authentication failed: {e}")
        return None


async def get_authorized_user(
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Require authenticated user"""
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    return current_user


# Health check endpoint
@app.get("/api/health", response_model=Dict[str, Any])
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "components": {
            "api": "operational",
            "strategy_manager": "operational",
            "backtest_engine": "operational",
            "websocket": "operational"
        }
    }


# System overview endpoint
@app.get("/api/system/overview")
async def get_system_overview(
    user: Dict[str, Any] = Depends(get_authorized_user)
):
    """Get comprehensive system overview"""
    try:
        # Get strategy statistics
        strategy_stats = await strategy_selector.get_strategy_statistics()
        
        # Get backtest engine status
        backtest_status = await backtest_engine.get_engine_status()
        
        # Get WebSocket connections
        ws_connections = {
            "total": websocket_manager.get_connection_count(),
            "strategy_specific": strategy_websocket_manager.get_connection_count()
        }
        
        return {
            "status": "operational",
            "timestamp": datetime.utcnow().isoformat(),
            "strategies": strategy_stats,
            "backtesting": backtest_status,
            "websocket": ws_connections,
            "user": {
                "id": user["id"],
                "name": user["name"],
                "role": user["role"]
            }
        }
        
    except Exception as e:
        logger.exception(f"Error getting system overview: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get system overview"
        )


# Error handling for unhandled exceptions
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.exception(f"Unhandled exception: {exc}")
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.environment == "development" else "An error occurred",
            "timestamp": datetime.utcnow().isoformat()
        }
    )


# Startup logging
@app.on_event("startup")
async def startup_event():
    """Log API startup"""
    logger.info("🌐 Trading Strategy API is ready")
    logger.info(f"📖 API Documentation: {settings.api_base_url}/api/docs")
    logger.info(f"🔗 API Redoc: {settings.api_base_url}/api/redoc")


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        log_level=settings.log_level.lower()
    )