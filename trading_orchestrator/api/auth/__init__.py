"""
Authentication module for API

Provides JWT-based authentication and user session management
"""

from .authentication import AuthenticationManager, User
from .authorization import AuthorizationManager, Permission, Role

__all__ = [
    "AuthenticationManager",
    "AuthorizationManager",
    "User",
    "Permission",
    "Role",
]
