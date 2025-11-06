"""
Authentication Manager for API access control

Provides authentication services:
- Token validation
- User session management
- JWT token handling
- Session security
"""

import asyncio
import hashlib
import secrets
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass

from fastapi import HTTPException, status
from loguru import logger


@dataclass
class User:
    """User model for authentication"""
    id: str
    name: str
    email: str
    role: str
    permissions: List[str]
    created_at: datetime
    last_login: Optional[datetime] = None
    is_active: bool = True
    token_hash: Optional[str] = None


@dataclass
class AuthToken:
    """Authentication token model"""
    token: str
    user_id: str
    expires_at: datetime
    created_at: datetime
    last_used: Optional[datetime] = None
    permissions: List[str]


class AuthenticationManager:
    """
    Authentication manager for handling user authentication
    
    Manages user sessions, token validation, and authentication
    """
    
    def __init__(self):
        self.users: Dict[str, User] = {}
        self.tokens: Dict[str, AuthToken] = {}
        self.token_expiry_hours = 24
        self.max_sessions_per_user = 5
        
        # Initialize with demo user for testing
        self._create_demo_user()
    
    def _create_demo_user(self):
        """Create demo user for testing"""
        demo_user = User(
            id="demo_user",
            name="Demo User",
            email="demo@example.com",
            role="trader",
            permissions=["read_strategies", "write_strategies", "execute_strategies", "view_performance"],
            created_at=datetime.utcnow()
        )
        
        # Generate demo token
        demo_token = secrets.token_urlsafe(32)
        demo_token_hash = hashlib.sha256(demo_token.encode()).hexdigest()
        demo_user.token_hash = demo_token_hash
        
        self.users[demo_user.id] = demo_user
        
        auth_token = AuthToken(
            token=demo_token_hash,
            user_id=demo_user.id,
            expires_at=datetime.utcnow() + timedelta(hours=self.token_expiry_hours),
            created_at=datetime.utcnow(),
            permissions=demo_user.permissions
        )
        
        self.tokens[demo_token_hash] = auth_token
        
        logger.info(f"Demo user created with token: {demo_token}")
    
    async def authenticate_user(self, username: str, password: str) -> Optional[str]:
        """
        Authenticate user with username and password
        
        Returns: Authentication token if successful, None otherwise
        """
        try:
            # Find user by username/email
            user = None
            for u in self.users.values():
                if (u.email.lower() == username.lower() or 
                    u.name.lower() == username.lower()):
                    user = u
                    break
            
            if not user or not user.is_active:
                logger.warning(f"Authentication failed for user: {username}")
                return None
            
            # Check password (in production, use proper password hashing)
            # For demo purposes, accept any password
            if password != "demo123":
                logger.warning(f"Invalid password for user: {username}")
                return None
            
            # Clean up expired tokens for this user
            await self._cleanup_user_tokens(user.id)
            
            # Check if user has too many active sessions
            active_sessions = len([t for t in self.tokens.values() 
                                 if t.user_id == user.id and t.expires_at > datetime.utcnow()])
            
            if active_sessions >= self.max_sessions_per_user:
                logger.warning(f"User {username} has exceeded maximum sessions")
                return None
            
            # Create new token
            token = secrets.token_urlsafe(32)
            token_hash = hashlib.sha256(token.encode()).hexdigest()
            
            auth_token = AuthToken(
                token=token_hash,
                user_id=user.id,
                expires_at=datetime.utcnow() + timedelta(hours=self.token_expiry_hours),
                created_at=datetime.utcnow(),
                permissions=user.permissions
            )
            
            self.tokens[token_hash] = auth_token
            
            # Update user last login
            user.last_login = datetime.utcnow()
            user.token_hash = token_hash
            
            logger.info(f"User {username} authenticated successfully")
            return token
            
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return None
    
    async def authenticate_token(self, token: str) -> Optional[User]:
        """
        Authenticate using token
        
        Returns: User object if token is valid, None otherwise
        """
        try:
            token_hash = hashlib.sha256(token.encode()).hexdigest()
            
            auth_token = self.tokens.get(token_hash)
            if not auth_token:
                return None
            
            # Check if token is expired
            if auth_token.expires_at <= datetime.utcnow():
                # Remove expired token
                del self.tokens[token_hash]
                logger.warning(f"Expired token used: {token_hash[:8]}...")
                return None
            
            # Get user
            user = self.users.get(auth_token.user_id)
            if not user or not user.is_active:
                return None
            
            # Update token last used
            auth_token.last_used = datetime.utcnow()
            
            return user
            
        except Exception as e:
            logger.error(f"Token authentication error: {e}")
            return None
    
    async def refresh_token(self, token: str) -> Optional[str]:
        """
        Refresh an authentication token
        
        Returns: New token if refresh successful, None otherwise
        """
        try:
            # Authenticate existing token
            user = await self.authenticate_token(token)
            if not user:
                return None
            
            # Get original token hash
            original_token_hash = None
            for th, at in self.tokens.items():
                if at.user_id == user.id and th == hashlib.sha256(token.encode()).hexdigest():
                    original_token_hash = th
                    break
            
            if not original_token_hash:
                return None
            
            # Remove old token
            if original_token_hash in self.tokens:
                del self.tokens[original_token_hash]
            
            # Create new token
            new_token = secrets.token_urlsafe(32)
            new_token_hash = hashlib.sha256(new_token.encode()).hexdigest()
            
            auth_token = AuthToken(
                token=new_token_hash,
                user_id=user.id,
                expires_at=datetime.utcnow() + timedelta(hours=self.token_expiry_hours),
                created_at=datetime.utcnow(),
                permissions=user.permissions
            )
            
            self.tokens[new_token_hash] = auth_token
            user.token_hash = new_token_hash
            
            logger.info(f"Token refreshed for user: {user.name}")
            return new_token
            
        except Exception as e:
            logger.error(f"Token refresh error: {e}")
            return None
    
    async def logout(self, token: str) -> bool:
        """
        Logout user by invalidating token
        
        Returns: True if logout successful
        """
        try:
            token_hash = hashlib.sha256(token.encode()).hexdigest()
            
            if token_hash in self.tokens:
                del self.tokens[token_hash]
                logger.info(f"Token invalidated: {token_hash[:8]}...")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Logout error: {e}")
            return False
    
    async def logout_all_sessions(self, user_id: str) -> int:
        """
        Logout all sessions for a user
        
        Returns: Number of sessions invalidated
        """
        try:
            invalidated_count = 0
            tokens_to_remove = []
            
            for token_hash, auth_token in self.tokens.items():
                if auth_token.user_id == user_id:
                    tokens_to_remove.append(token_hash)
            
            for token_hash in tokens_to_remove:
                del self.tokens[token_hash]
                invalidated_count += 1
            
            logger.info(f"Invalidated {invalidated_count} sessions for user: {user_id}")
            return invalidated_count
            
        except Exception as e:
            logger.error(f"Logout all sessions error: {e}")
            return 0
    
    async def get_user_permissions(self, token: str) -> List[str]:
        """
        Get user permissions for a token
        
        Returns: List of permissions
        """
        try:
            token_hash = hashlib.sha256(token.encode()).hexdigest()
            auth_token = self.tokens.get(token_hash)
            
            if not auth_token or auth_token.expires_at <= datetime.utcnow():
                return []
            
            return auth_token.permissions
            
        except Exception as e:
            logger.error(f"Get permissions error: {e}")
            return []
    
    async def has_permission(self, token: str, permission: str) -> bool:
        """
        Check if user has specific permission
        
        Returns: True if user has permission
        """
        try:
            permissions = await self.get_user_permissions(token)
            return permission in permissions
            
        except Exception as e:
            logger.error(f"Permission check error: {e}")
            return False
    
    async def _cleanup_user_tokens(self, user_id: str):
        """Clean up expired tokens for a user"""
        try:
            tokens_to_remove = []
            
            for token_hash, auth_token in self.tokens.items():
                if (auth_token.user_id == user_id and 
                    auth_token.expires_at <= datetime.utcnow()):
                    tokens_to_remove.append(token_hash)
            
            for token_hash in tokens_to_remove:
                del self.tokens[token_hash]
                
        except Exception as e:
            logger.error(f"Token cleanup error: {e}")
    
    async def cleanup_expired_tokens(self):
        """Clean up all expired tokens"""
        try:
            tokens_to_remove = []
            
            for token_hash, auth_token in self.tokens.items():
                if auth_token.expires_at <= datetime.utcnow():
                    tokens_to_remove.append(token_hash)
            
            for token_hash in tokens_to_remove:
                del self.tokens[token_hash]
            
            if tokens_to_remove:
                logger.info(f"Cleaned up {len(tokens_to_remove)} expired tokens")
                
        except Exception as e:
            logger.error(f"Token cleanup error: {e}")
    
    def get_active_sessions_count(self, user_id: str) -> int:
        """Get number of active sessions for user"""
        try:
            return len([t for t in self.tokens.values() 
                       if t.user_id == user_id and t.expires_at > datetime.utcnow()])
        except Exception:
            return 0
    
    def get_all_users(self) -> List[User]:
        """Get all users (for admin purposes)"""
        return list(self.users.values())
    
    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        return self.users.get(user_id)
