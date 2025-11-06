"""
Authorization manager for API

Provides role-based access control (RBAC) and permission validation
"""

from enum import Enum
from typing import Dict, List, Set, Any, Optional
from datetime import datetime
from loguru import logger


class Permission(str, Enum):
    """Available permissions"""
    # Strategy permissions
    READ_STRATEGIES = "read_strategies"
    WRITE_STRATEGIES = "write_strategies"
    EXECUTE_STRATEGIES = "execute_strategies"
    DELETE_STRATEGIES = "delete_strategies"
    MANAGE_STRATEGIES = "manage_strategies"
    
    # Performance and analytics
    VIEW_PERFORMANCE = "view_performance"
    MANAGE_PERFORMANCE = "manage_performance"
    
    # Backtesting
    RUN_BACKTESTS = "run_backtests"
    VIEW_BACKTEST_RESULTS = "view_backtest_results"
    MANAGE_BACKTESTS = "manage_backtests"
    
    # User management
    MANAGE_USERS = "manage_users"
    VIEW_USER_ACTIVITIES = "view_user_activities"
    
    # System management
    MANAGE_SYSTEM = "manage_system"
    VIEW_SYSTEM_LOGS = "view_system_logs"
    CONFIGURE_SYSTEM = "configure_system"
    
    # WebSocket access
    USE_WEBSOCKETS = "use_websockets"
    MANAGE_WEBSOCKETS = "manage_websockets"
    
    # Data export/import
    EXPORT_DATA = "export_data"
    IMPORT_DATA = "import_data"
    
    # Alerts and notifications
    MANAGE_ALERTS = "manage_alerts"
    VIEW_ALERTS = "view_alerts"


class Role(str, Enum):
    """Available user roles"""
    ADMIN = "admin"
    TRADER = "trader"
    VIEWER = "viewer"
    ANALYST = "analyst"
    

# Role-based permission mappings
ROLE_PERMISSIONS = {
    Role.ADMIN: {
        Permission.READ_STRATEGIES,
        Permission.WRITE_STRATEGIES,
        Permission.EXECUTE_STRATEGIES,
        Permission.DELETE_STRATEGIES,
        Permission.MANAGE_STRATEGIES,
        Permission.VIEW_PERFORMANCE,
        Permission.MANAGE_PERFORMANCE,
        Permission.RUN_BACKTESTS,
        Permission.VIEW_BACKTEST_RESULTS,
        Permission.MANAGE_BACKTESTS,
        Permission.MANAGE_USERS,
        Permission.VIEW_USER_ACTIVITIES,
        Permission.MANAGE_SYSTEM,
        Permission.VIEW_SYSTEM_LOGS,
        Permission.CONFIGURE_SYSTEM,
        Permission.USE_WEBSOCKETS,
        Permission.MANAGE_WEBSOCKETS,
        Permission.EXPORT_DATA,
        Permission.IMPORT_DATA,
        Permission.MANAGE_ALERTS,
        Permission.VIEW_ALERTS,
    },
    Role.TRADER: {
        Permission.READ_STRATEGIES,
        Permission.WRITE_STRATEGIES,
        Permission.EXECUTE_STRATEGIES,
        Permission.DELETE_STRATEGIES,
        Permission.MANAGE_STRATEGIES,
        Permission.VIEW_PERFORMANCE,
        Permission.RUN_BACKTESTS,
        Permission.VIEW_BACKTEST_RESULTS,
        Permission.USE_WEBSOCKETS,
        Permission.VIEW_ALERTS,
    },
    Role.ANALYST: {
        Permission.READ_STRATEGIES,
        Permission.VIEW_PERFORMANCE,
        Permission.MANAGE_PERFORMANCE,
        Permission.RUN_BACKTESTS,
        Permission.VIEW_BACKTEST_RESULTS,
        Permission.USE_WEBSOCKETS,
        Permission.EXPORT_DATA,
        Permission.VIEW_ALERTS,
    },
    Role.VIEWER: {
        Permission.READ_STRATEGIES,
        Permission.VIEW_PERFORMANCE,
        Permission.VIEW_BACKTEST_RESULTS,
        Permission.USE_WEBSOCKETS,
        Permission.VIEW_ALERTS,
    },
}


class AuthorizationManager:
    """Manages role-based access control and permission validation"""
    
    def __init__(self):
        # User role assignments: user_id -> set of roles
        self.user_roles: Dict[str, Set[Role]] = {}
        
        # User-specific permission overrides: user_id -> set of permissions
        self.user_permissions: Dict[str, Set[Permission]] = {}
        
        # Resource access control: resource_type -> {resource_id -> set of user_ids}
        self.resource_access: Dict[str, Dict[str, Set[str]]] = {
            "strategy": {},
            "user": {},
            "backtest": {}
        }
        
        # Permission validation cache
        self._permission_cache: Dict[str, bool] = {}
        
        # Initialize default admin user
        self._setup_default_admin()
    
    def _setup_default_admin(self):
        """Setup default admin user"""
        admin_user_id = "admin"
        self.user_roles[admin_user_id] = {Role.ADMIN}
        logger.info(f"Default admin user created: {admin_user_id}")
    
    def assign_role(self, user_id: str, role: Role) -> bool:
        """Assign a role to a user"""
        try:
            if user_id not in self.user_roles:
                self.user_roles[user_id] = set()
            
            self.user_roles[user_id].add(role)
            self._clear_permission_cache(user_id)
            
            logger.info(f"Role {role.value} assigned to user: {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error assigning role to user {user_id}: {e}")
            return False
    
    def remove_role(self, user_id: str, role: Role) -> bool:
        """Remove a role from a user"""
        try:
            if user_id in self.user_roles and role in self.user_roles[user_id]:
                self.user_roles[user_id].remove(role)
                
                # Remove user if no roles left
                if not self.user_roles[user_id]:
                    del self.user_roles[user_id]
                
                self._clear_permission_cache(user_id)
                
                logger.info(f"Role {role.value} removed from user: {user_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error removing role from user {user_id}: {e}")
            return False
    
    def grant_permission(self, user_id: str, permission: Permission) -> bool:
        """Grant specific permission to user"""
        try:
            if user_id not in self.user_permissions:
                self.user_permissions[user_id] = set()
            
            self.user_permissions[user_id].add(permission)
            self._clear_permission_cache(user_id)
            
            logger.info(f"Permission {permission.value} granted to user: {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error granting permission to user {user_id}: {e}")
            return False
    
    def revoke_permission(self, user_id: str, permission: Permission) -> bool:
        """Revoke specific permission from user"""
        try:
            if user_id in self.user_permissions and permission in self.user_permissions[user_id]:
                self.user_permissions[user_id].remove(permission)
                
                # Remove user if no permissions left
                if not self.user_permissions[user_id]:
                    del self.user_permissions[user_id]
                
                self._clear_permission_cache(user_id)
                
                logger.info(f"Permission {permission.value} revoked from user: {user_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error revoking permission from user {user_id}: {e}")
            return False
    
    def has_permission(self, user_id: str, permission: Permission) -> bool:
        """Check if user has specific permission"""
        try:
            cache_key = f"{user_id}:{permission.value}"
            
            # Check cache first
            if cache_key in self._permission_cache:
                return self._permission_cache[cache_key]
            
            # Get user's roles and explicit permissions
            roles = self.user_roles.get(user_id, set())
            explicit_permissions = self.user_permissions.get(user_id, set())
            
            # Check explicit permissions first
            if permission in explicit_permissions:
                self._permission_cache[cache_key] = True
                return True
            
            # Check role-based permissions
            for role in roles:
                role_permissions = ROLE_PERMISSIONS.get(role, set())
                if permission in role_permissions:
                    self._permission_cache[cache_key] = True
                    return True
            
            # No permission found
            self._permission_cache[cache_key] = False
            return False
            
        except Exception as e:
            logger.error(f"Error checking permission for user {user_id}: {e}")
            return False
    
    def has_any_permission(self, user_id: str, permissions: List[Permission]) -> bool:
        """Check if user has any of the specified permissions"""
        for permission in permissions:
            if self.has_permission(user_id, permission):
                return True
        return False
    
    def has_all_permissions(self, user_id: str, permissions: List[Permission]) -> bool:
        """Check if user has all of the specified permissions"""
        for permission in permissions:
            if not self.has_permission(user_id, permission):
                return False
        return True
    
    def get_user_permissions(self, user_id: str) -> Set[Permission]:
        """Get all permissions for a user"""
        try:
            roles = self.user_roles.get(user_id, set())
            explicit_permissions = self.user_permissions.get(user_id, set())
            
            # Collect all role-based permissions
            role_permissions = set()
            for role in roles:
                role_permissions.update(ROLE_PERMISSIONS.get(role, set()))
            
            # Combine with explicit permissions
            all_permissions = role_permissions.union(explicit_permissions)
            
            return all_permissions
            
        except Exception as e:
            logger.error(f"Error getting permissions for user {user_id}: {e}")
            return set()
    
    def get_user_roles(self, user_id: str) -> Set[Role]:
        """Get all roles for a user"""
        return self.user_roles.get(user_id, set())
    
    def grant_resource_access(
        self,
        resource_type: str,
        resource_id: str,
        user_id: str,
        permission: Permission
    ) -> bool:
        """Grant user access to specific resource"""
        try:
            if resource_type not in self.resource_access:
                self.resource_access[resource_type] = {}
            
            if resource_id not in self.resource_access[resource_type]:
                self.resource_access[resource_type][resource_id] = set()
            
            # Map permissions to user access
            access_key = f"{user_id}:{permission.value}"
            self.resource_access[resource_type][resource_id].add(access_key)
            
            logger.info(f"Resource access granted: {resource_type}:{resource_id} to user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error granting resource access: {e}")
            return False
    
    def revoke_resource_access(
        self,
        resource_type: str,
        resource_id: str,
        user_id: str,
        permission: Permission
    ) -> bool:
        """Revoke user access to specific resource"""
        try:
            if (resource_type in self.resource_access and 
                resource_id in self.resource_access[resource_type]):
                
                access_key = f"{user_id}:{permission.value}"
                self.resource_access[resource_type][resource_id].discard(access_key)
                
                # Clean up empty sets
                if not self.resource_access[resource_type][resource_id]:
                    del self.resource_access[resource_type][resource_id]
                
                logger.info(f"Resource access revoked: {resource_type}:{resource_id} from user {user_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error revoking resource access: {e}")
            return False
    
    def has_resource_access(
        self,
        resource_type: str,
        resource_id: str,
        user_id: str,
        permission: Permission
    ) -> bool:
        """Check if user has access to specific resource"""
        try:
            access_key = f"{user_id}:{permission.value}"
            
            return (
                resource_type in self.resource_access and
                resource_id in self.resource_access[resource_type] and
                access_key in self.resource_access[resource_type][resource_id]
            )
            
        except Exception as e:
            logger.error(f"Error checking resource access: {e}")
            return False
    
    def get_users_with_resource_access(
        self,
        resource_type: str,
        resource_id: str
    ) -> List[str]:
        """Get all users with access to specific resource"""
        try:
            if (resource_type in self.resource_access and 
                resource_id in self.resource_access[resource_type]):
                
                access_keys = self.resource_access[resource_type][resource_id]
                users = [key.split(":")[0] for key in access_keys]
                return list(set(users))  # Remove duplicates
            
            return []
            
        except Exception as e:
            logger.error(f"Error getting users with resource access: {e}")
            return []
    
    def _clear_permission_cache(self, user_id: str):
        """Clear permission cache for user"""
        keys_to_remove = [key for key in self._permission_cache.keys() if key.startswith(f"{user_id}:")]
        for key in keys_to_remove:
            del self._permission_cache[key]
    
    def get_all_users(self) -> List[str]:
        """Get all users with roles"""
        return list(self.user_roles.keys())
    
    def get_users_by_role(self, role: Role) -> List[str]:
        """Get all users with specific role"""
        return [user_id for user_id, roles in self.user_roles.items() if role in roles]
    
    def get_role_statistics(self) -> Dict[str, int]:
        """Get statistics about role distribution"""
        stats = {}
        for role in Role:
            stats[role.value] = len(self.get_users_by_role(role))
        return stats
    
    def validate_user_context(
        self,
        user_id: str,
        required_permissions: List[Permission],
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Validate user has required permissions and resource access"""
        result = {
            "valid": True,
            "missing_permissions": [],
            "resource_access_denied": False,
            "user_roles": [],
            "user_permissions": []
        }
        
        # Get user information
        user_roles = self.get_user_roles(user_id)
        user_permissions = self.get_user_permissions(user_id)
        
        result["user_roles"] = [role.value for role in user_roles]
        result["user_permissions"] = [perm.value for perm in user_permissions]
        
        # Check permissions
        for permission in required_permissions:
            if not self.has_permission(user_id, permission):
                result["missing_permissions"].append(permission.value)
                result["valid"] = False
        
        # Check resource access if specified
        if resource_type and resource_id:
            for permission in required_permissions:
                if not self.has_resource_access(resource_type, resource_id, user_id, permission):
                    result["resource_access_denied"] = True
                    result["valid"] = False
        
        return result
    
    def get_authorization_summary(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive authorization summary for user"""
        roles = self.get_user_roles(user_id)
        permissions = self.get_user_permissions(user_id)
        
        # Get resource access summary
        resource_access = {}
        for resource_type, resources in self.resource_access.items():
            resource_access[resource_type] = {}
            for resource_id, access_keys in resources.items():
                user_access_keys = [key for key in access_keys if key.startswith(f"{user_id}:")]
                if user_access_keys:
                    resource_access[resource_type][resource_id] = [
                        key.split(":")[1] for key in user_access_keys
                    ]
        
        return {
            "user_id": user_id,
            "roles": [role.value for role in roles],
            "permissions": [perm.value for perm in permissions],
            "role_permissions": sum(1 for _ in permissions),  # Simplified count
            "explicit_permissions": len([p for p in permissions if 
                                       not any(p in ROLE_PERMISSIONS.get(r, set()) for r in roles)]),
            "resource_access": resource_access,
            "created_at": datetime.utcnow().isoformat()
        }
