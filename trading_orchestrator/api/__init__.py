"""
API Module for Trading Orchestrator Strategy Library

This module provides REST API endpoints and WebSocket connections for:
- Strategy management (CRUD operations)
- Real-time strategy monitoring
- Backtesting and performance analysis
- Strategy comparison and selection
"""

from .main import app
from .schemas.strategies import (
    StrategyConfig,
    StrategyResponse,
    StrategyCreateRequest,
    StrategyUpdateRequest,
    StrategyPerformance,
    StrategySignal,
    BacktestRequest,
    BacktestResponse,
    StrategyComparison,
    StrategyFilter,
    StrategyCategory,
    StrategyStatus,
    SignalType,
    RiskLevel,
    APIResponse,
    PaginatedResponse,
    WebSocketMessage
)

from .routers.strategies import router as strategies_router
from .websocket.router import router as websocket_router

__all__ = [
    'app',
    'strategies_router',
    'websocket_router',
    'StrategyConfig',
    'StrategyResponse',
    'StrategyCreateRequest',
    'StrategyUpdateRequest', 
    'StrategyPerformance',
    'StrategySignal',
    'BacktestRequest',
    'BacktestResponse',
    'StrategyComparison',
    'StrategyFilter',
    'StrategyCategory',
    'StrategyStatus',
    'SignalType',
    'RiskLevel',
    'APIResponse',
    'PaginatedResponse',
    'WebSocketMessage'
]