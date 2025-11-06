"""
Dependencies for API routers

Provides common dependencies for:
- Strategy selector access
- Backtest engine access
- WebSocket manager access
- User authentication and authorization
- Strategy access validation
"""

from typing import Dict, List, Optional, Any
from fastapi import HTTPException, status

from ..auth.authentication import AuthenticationManager
from ..auth.authorization import AuthorizationManager
from ...strategies.strategy_management import StrategySelector
from ...strategies.enhanced_backtesting import EnhancedBacktestEngine
from ...api.websocket.strategy_websocket import StrategyWebSocketManager


# Global instances
_auth_manager = None
_authz_manager = None
_strategy_selector = None
_backtest_engine = None
_strategy_websocket_manager = None


async def get_authentication_manager() -> AuthenticationManager:
    """Get authentication manager instance"""
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = AuthenticationManager()
    return _auth_manager


async def get_authorization_manager() -> AuthorizationManager:
    """Get authorization manager instance"""
    global _authz_manager
    if _authz_manager is None:
        _authz_manager = AuthorizationManager()
    return _authz_manager


async def get_strategy_selector() -> StrategySelector:
    """Get strategy selector instance"""
    global _strategy_selector
    if _strategy_selector is None:
        _strategy_selector = StrategySelector()
    return _strategy_selector


async def get_backtest_engine() -> EnhancedBacktestEngine:
    """Get backtest engine instance"""
    global _backtest_engine
    if _backtest_engine is None:
        _backtest_engine = EnhancedBacktestEngine()
    return _backtest_engine


async def get_strategy_websocket_manager() -> StrategyWebSocketManager:
    """Get strategy WebSocket manager instance"""
    global _strategy_websocket_manager
    if _strategy_websocket_manager is None:
        _strategy_websocket_manager = StrategyWebSocketManager()
    return _strategy_websocket_manager


async def get_current_user(
    auth_manager: AuthenticationManager = None
) -> Dict[str, Any]:
    """Get current authenticated user (placeholder implementation)"""
    # This would integrate with your actual authentication system
    # For now, return a mock user for demonstration
    return {
        "id": "user_123",
        "name": "Demo User",
        "email": "demo@example.com",
        "role": "trader",
        "permissions": ["read_strategies", "write_strategies", "execute_strategies"]
    }


async def validate_strategy_access(strategy_id: str, user_id: str) -> bool:
    """Validate that user has access to strategy"""
    try:
        # This would check actual access rights
        # For now, assume all users have access to all strategies
        return True
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this strategy"
        )


async def validate_user_permissions(
    user: Dict[str, Any],
    required_permissions: List[str]
) -> bool:
    """Validate user has required permissions"""
    user_permissions = user.get("permissions", [])
    
    for permission in required_permissions:
        if permission not in user_permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission required: {permission}"
            )
    
    return True


async def validate_strategy_ownership(strategy_id: str, user_id: str) -> bool:
    """Validate that user owns the strategy"""
    try:
        strategy_selector = await get_strategy_selector()
        strategy = await strategy_selector.get_strategy(strategy_id)
        
        if not strategy or strategy.get("created_by") != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only strategy owners can perform this action"
            )
        
        return True
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Failed to validate strategy ownership"
        )
