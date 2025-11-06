"""
Main FastAPI application for Trading Orchestrator Strategy API

Provides REST API endpoints and WebSocket connections for:
- Strategy management and execution
- Performance analytics and backtesting
- Real-time strategy monitoring
"""

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from .routers import strategies, websocket
from .auth import authentication, authorization
from .utils import error_handlers, response_formatter
from .utils.json_encoder import JSONEncoder


# Create FastAPI application
app = FastAPI(
    title="Trading Orchestrator Strategy API",
    description="Comprehensive API for trading strategy management, backtesting, and real-time monitoring",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    default_response_class=JSONResponse
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add trusted host middleware for security
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "127.0.0.1", "*.example.com"]  # Configure for production
)

# Setup error handlers
error_handlers.setup_error_handlers(app)

# Include routers
app.include_router(strategies.router, prefix="/api")
app.include_router(websocket.router, prefix="/api")


@app.get("/api/health", tags=["health"])
async def health_check():
    """Health check endpoint"""
    return response_formatter.ResponseFormatter.success_response(
        data={
            "status": "healthy",
            "service": "Trading Orchestrator Strategy API",
            "version": "1.0.0"
        },
        message="Service is healthy"
    )


@app.get("/api/system/overview", tags=["system"])
async def system_overview():
    """Get system overview and statistics"""
    return response_formatter.ResponseFormatter.success_response(
        data={
            "total_strategies": 0,
            "active_strategies": 0,
            "total_trades_today": 0,
            "total_portfolio_value": 0.0,
            "daily_return": 0.0,
            "system_uptime": "0d 0h 0m",
            "connected_websockets": 0
        },
        message="System overview retrieved successfully"
    )


@app.get("/api/metrics", tags=["monitoring"])
async def get_metrics():
    """Get application metrics for monitoring"""
    # This would integrate with your monitoring system
    return {
        "api_requests_total": 0,
        "api_request_duration_seconds": {
            "avg": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0
        },
        "websocket_connections_active": 0,
        "strategies_active": 0,
        "backtests_running": 0
    }


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("Trading Orchestrator Strategy API starting up...")
    
    # Initialize any background services
    # This would include strategy monitoring, WebSocket managers, etc.
    
    logger.info("API startup completed")


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown"""
    logger.info("Trading Orchestrator Strategy API shutting down...")
    
    # Cleanup any background services
    # Close WebSocket connections, stop monitoring tasks, etc.
    
    logger.info("API shutdown completed")


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Trading Orchestrator Strategy API",
        "version": "1.0.0",
        "status": "running",
        "documentation": "/api/docs",
        "health": "/api/health"
    }


# Configure logging
if __name__ != "__main__":
    # Remove default logger to avoid duplicate logging
    logger.remove()
    
    # Add custom format for production
    logger.add(
        "logs/api_{time}.log",
        rotation="1 day",
        retention="30 days",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}"
    )
    
    # Console output for development
    logger.add(
        lambda msg: print(msg, end=""),
        level="DEBUG",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>"
    )
