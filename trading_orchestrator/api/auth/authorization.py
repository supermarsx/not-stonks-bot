"""
Authorization Manager for API access control

Provides authorization services:
- Role-based access control (RBAC)
- Resource permission validation
- Strategy access control
- API endpoint protection
"""

from typing import Dict, List, Optional, Any, Set
from enum import Enum
from datetime import datetime

from fastapi import HTTPException, status
from loguru import logger


class Permission(str, Enum):
    """API permissions"""
    # Strategy permissions
    READ_STRATEGIES = "read_strategies"
    WRITE_STRATEGIES = "write_strategies"
    EXECUTE_STRATEGIES = "execute_strategies"
    DELETE_STRATEGIES = "delete_strategies"
    VIEW_PERFORMANCE = "view_performance"
    RUN_BACKTESTS = "run_backtests"
    MANAGE_ENSEMBLES = "manage_ensembles"
    
    # User management
    MANAGE_USERS = "manage_users"
    
    # System administration
    SYSTEM_ADMIN = "system_admin"
    
    # Trade execution
    PLACE_TRADES = "place_trades"
    VIEW_TRADES = "view_trades"
    
    # Risk management
    VIEW_RISK = "view_risk"
    MANAGE_RISK = "manage_risk"


class Role(str, Enum):
    """User roles"""
    ADMIN = "admin"
    TRADER = "trader"
    ANALYST = "analyst"
    VIEWER = "viewer"


class ResourceType(str, Enum):
    """Resource types for authorization"""
    STRATEGY = "strategy"
    PORTFOLIO = "portfolio"
    TRADE = "trade"
    USER = "user"
    SYSTEM = "system"


class AuthorizationManager:
    """
    Authorization manager for role-based access control
    
    Manages user roles, permissions, and resource access
    """
    
    def __init__(self):
        # Role permissions mapping
        self.role_permissions: Dict[Role, Set[Permission]] = {
            Role.ADMIN: {
                Permission.READ_STRATEGIES, Permission.WRITE_STRATEGIES,
                Permission.EXECUTE_STRATEGIES, Permission.DELETE_STRATEGIES,
                Permission.VIEW_PERFORMANCE, Permission.RUN_BACKTESTS,
                Permission.MANAGE_ENSEMBLES, Permission.MANAGE_USERS,
                Permission.SYSTEM_ADMIN, Permission.PLACE_TRADES,
                Permission.VIEW_TRADES, Permission.VIEW_RISK, Permission.MANAGE_RISK
            },
            Role.TRADER: {
                Permission.READ_STRATEGIES, Permission.WRITE_STRATEGIES,
                Permission.EXECUTE_STRATEGIES, Permission.VIEW_PERFORMANCE,
                Permission.RUN_BACKTESTS, Permission.MANAGE_ENSEMBLES,
                Permission.PLACE_TRADES, Permission.VIEW_TRADES,
                Permission.VIEW_RISK
            },
            Role.ANALYST: {
                Permission.READ_STRATEGIES, Permission.VIEW_PERFORMANCE,
                Permission.RUN_BACKTESTS, Permission.VIEW_TRADES,
                Permission.VIEW_RISK
            },
            Role.VIEWER: {
                Permission.READ_STRATEGIES, Permission.VIEW_PERFORMANCE
            }
        }
        
        # Resource ownership cache
        self.resource_ownership: Dict[str, Dict[str, Any]] = {}
    
    def get_user_permissions(self, user: Dict[str, Any]) -> Set[Permission]:
        """
        Get permissions for a user based on their role
        
        Returns: Set of permissions
        """
        try:
            role_str = user.get("role", "viewer").lower()
            role = Role(role_str) if role_str in [r.value for r in Role] else Role.VIEWER
            return self.role_permissions.get(role, self.role_permissions[Role.VIEWER])
            
        except Exception as e:
            logger.error(f"Error getting user permissions: {e}")
            return self.role_permissions[Role.VIEWER]
    
    def has_permission(self, user: Dict[str, Any], permission: Permission) -> bool:
        """
        Check if user has specific permission
        
        Returns: True if user has permission
        """
        try:
            user_permissions = self.get_user_permissions(user)
            return permission in user_permissions
            
        except Exception as e:
            logger.error(f"Error checking permission: {e}")
            return False
    
    def has_any_permission(self, user: Dict[str, Any], permissions: List[Permission]) -> bool:
        """
        Check if user has any of the specified permissions
        
        Returns: True if user has at least one permission
        """
        try:
            user_permissions = self.get_user_permissions(user)
            return any(perm in user_permissions for perm in permissions)
            
        except Exception as e:
            logger.error(f"Error checking permissions: {e}")
            return False
    
    def has_all_permissions(self, user: Dict[str, Any], permissions: List[Permission]) -> bool:
        """
        Check if user has all of the specified permissions
        
        Returns: True if user has all permissions
        """
        try:
            user_permissions = self.get_user_permissions(user)
            return all(perm in user_permissions for perm in permissions)
            
        except Exception as e:
            logger.error(f"Error checking permissions: {e}")
            return False
    
    def require_permission(self, user: Dict[str, Any], permission: Permission):
        """
        Raise HTTPException if user doesn't have permission
        
        Raises: HTTPException with 403 status if permission denied
        """
        if not self.has_permission(user, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission required: {permission.value}"
            )
    
    def require_any_permission(self, user: Dict[str, Any], permissions: List[Permission]):
        """
        Raise HTTPException if user doesn't have any of the permissions
        
        Raises: HTTPException with 403 status if permission denied
        """
        if not self.has_any_permission(user, permissions):
            permission_names = [perm.value for perm in permissions]
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"At least one permission required: {', '.join(permission_names)}"
            )
    
    def require_all_permissions(self, user: Dict[str, Any], permissions: List[Permission]):
        """
        Raise HTTPException if user doesn't have all of the permissions
        
        Raises: HTTPException with 403 status if permission denied
        """
        if not self.has_all_permissions(user, permissions):
            permission_names = [perm.value for perm in permissions]
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"All permissions required: {', '.join(permission_names)}"
            )
    
    def can_access_strategy(self, user: Dict[str, Any], strategy_id: str, owner_id: str) -> bool:
        """
        Check if user can access a strategy
        
        Returns: True if user can access strategy
        """
        try:
            # Admin can access all strategies
            if self.has_permission(user, Permission.SYSTEM_ADMIN):
                return True
            
            # Strategy owner can always access
            if user.get("id") == owner_id:
                return True
            
            # Check if strategy is shared (this would come from database)
            # For now, assume private strategies only
            return False
            
        except Exception as e:
            logger.error(f"Error checking strategy access: {e}")
            return False
    
    def require_strategy_access(self, user: Dict[str, Any], strategy_id: str, owner_id: str):
        """
        Raise HTTPException if user can't access strategy
        
        Raises: HTTPException with 403 status if access denied
        """
        if not self.can_access_strategy(user, strategy_id, owner_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this strategy"
            )
    
    def can_modify_strategy(self, user: Dict[str, Any], strategy_id: str, owner_id: str) -> bool:
        """
        Check if user can modify a strategy
        
        Returns: True if user can modify strategy
        """
        try:
            # Admin can modify all strategies
            if self.has_permission(user, Permission.SYSTEM_ADMIN):
                return True
            
            # Strategy owner can modify
            if user.get("id") == owner_id:
                return True
            
            # Check write permission for non-owners (shared strategies)
            return self.has_permission(user, Permission.WRITE_STRATEGIES)
            
        except Exception as e:
            logger.error(f"Error checking strategy modification access: {e}")
            return False
    
    def require_strategy_modification(self, user: Dict[str, Any], strategy_id: str, owner_id: str):
        """
        Raise HTTPException if user can't modify strategy
        
        Raises: HTTPException with 403 status if modification denied
        """
        if not self.can_modify_strategy(user, strategy_id, owner_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Permission denied to modify this strategy"
            )
    
    def can_execute_strategy(self, user: Dict[str, Any], strategy_id: str, owner_id: str) -> bool:
        """
        Check if user can execute a strategy
        
        Returns: True if user can execute strategy
        """
        try:
            # Admin can execute all strategies
            if self.has_permission(user, Permission.SYSTEM_ADMIN):
                return True
            
            # Strategy owner can execute
            if user.get("id") == owner_id:
                return True
            
            # Check execute permission for shared strategies
            return self.has_permission(user, Permission.EXECUTE_STRATEGIES)
            
        except Exception as e:
            logger.error(f"Error checking strategy execution access: {e}")
            return False
    
    def require_strategy_execution(self, user: Dict[str, Any], strategy_id: str, owner_id: str):
        """
        Raise HTTPException if user can't execute strategy
        
        Raises: HTTPException with 403 status if execution denied
        """
        if not self.can_execute_strategy(user, strategy_id, owner_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Permission denied to execute this strategy"
            )
    
    def can_delete_strategy(self, user: Dict[str, Any], strategy_id: str, owner_id: str) -> bool:
        """
        Check if user can delete a strategy
        
        Returns: True if user can delete strategy
        """
        try:
            # Only strategy owner or admin can delete
            if user.get("id") == owner_id:
                return True
            
            return self.has_permission(user, Permission.SYSTEM_ADMIN)
            
        except Exception as e:
            logger.error(f"Error checking strategy deletion access: {e}")
            return False
    
    def require_strategy_deletion(self, user: Dict[str, Any], strategy_id: str, owner_id: str):
        """
        Raise HTTPException if user can't delete strategy
        
        Raises: HTTPException with 403 status if deletion denied
        """
        if not self.can_delete_strategy(user, strategy_id, owner_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only strategy owners or administrators can delete strategies"
            )
    
    def can_manage_users(self, user: Dict[str, Any]) -> bool:
        """Check if user can manage other users"""
        return self.has_permission(user, Permission.MANAGE_USERS)
    
    def require_user_management(self, user: Dict[str, Any]):
        """Raise HTTPException if user can't manage users"""
        if not self.can_manage_users(user):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Permission denied to manage users"
            )
    
    def can_access_system_admin(self, user: Dict[str, Any]) -> bool:
        """Check if user can access system administration features"""
        return self.has_permission(user, Permission.SYSTEM_ADMIN)
    
    def require_system_admin(self, user: Dict[str, Any]):
        """Raise HTTPException if user can't access system administration"""
        if not self.can_access_system_admin(user):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="System administration access required"
            )
    
    def get_user_role(self, user: Dict[str, Any]) -> Role:
        """Get user role"""
        try:
            role_str = user.get("role", "viewer").lower()
            return Role(role_str) if role_str in [r.value for r in Role] else Role.VIEWER
        except Exception as e:
            logger.error(f"Error getting user role: {e}")
            return Role.VIEWER
    
    def is_admin(self, user: Dict[str, Any]) -> bool:
        """Check if user is an administrator"""
        return self.get_user_role(user) == Role.ADMIN
    
    def is_trader(self, user: Dict[str, Any]) -> bool:
        """Check if user is a trader"""
        return self.get_user_role(user) in [Role.ADMIN, Role.TRADER]
    
    def can_place_trades(self, user: Dict[str, Any]) -> bool:
        """Check if user can place trades"""
        return self.has_permission(user, Permission.PLACE_TRADES)
    
    def require_trade_permission(self, user: Dict[str, Any]):
        """Raise HTTPException if user can't place trades"""
        if not self.can_place_trades(user):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Permission denied to place trades"
            )
