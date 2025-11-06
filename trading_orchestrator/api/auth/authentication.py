"""
Authentication manager for API

Provides JWT token validation, user session management,
and password hashing functionality.
"""

import jwt
import bcrypt
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from loguru import logger


class User:
    """User model for authentication"""
    
    def __init__(
        self,
        id: str,
        name: str,
        email: str,
        password_hash: str,
        role: str = "trader",
        permissions: Optional[List[str]] = None,
        is_active: bool = True
    ):
        self.id = id
        self.name = name
        self.email = email
        self.password_hash = password_hash
        self.role = role
        self.permissions = permissions or []
        self.is_active = is_active
        self.created_at = datetime.utcnow()
        self.last_login = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert user to dictionary (excluding password)"""
        return {
            "id": self.id,
            "name": self.name,
            "email": self.email,
            "role": self.role,
            "permissions": self.permissions,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
            "last_login": self.last_login.isoformat() if self.last_login else None
        }
    
    def verify_password(self, password: str) -> bool:
        """Verify user password"""
        try:
            return bcrypt.checkpw(
                password.encode('utf-8'),
                self.password_hash.encode('utf-8')
            )
        except Exception as e:
            logger.error(f"Password verification error: {e}")
            return False


class AuthenticationManager:
    """Manages user authentication and JWT token operations"""
    
    def __init__(self, secret_key: Optional[str] = None):
        self.secret_key = secret_key or "your-secret-key-here"
        self.algorithm = "HS256"
        self.token_expiration_hours = 24
        self._users_cache: Dict[str, User] = {}
        self._active_sessions: Dict[str, Dict[str, Any]] = {}
    
    def create_user(
        self,
        name: str,
        email: str,
        password: str,
        role: str = "trader",
        permissions: Optional[List[str]] = None
    ) -> User:
        """Create a new user"""
        user_id = f"user_{len(self._users_cache) + 1}"
        password_hash = bcrypt.hashpw(
            password.encode('utf-8'),
            bcrypt.gensalt()
        ).decode('utf-8')
        
        user = User(
            id=user_id,
            name=name,
            email=email,
            password_hash=password_hash,
            role=role,
            permissions=permissions
        )
        
        self._users_cache[user_id] = user
        logger.info(f"User created: {user_id} ({email})")
        
        return user
    
    def authenticate_user(self, email: str, password: str) -> Optional[User]:
        """Authenticate user with email and password"""
        for user in self._users_cache.values():
            if user.email.lower() == email.lower() and user.verify_password(password):
                if not user.is_active:
                    logger.warning(f"Inactive user attempted login: {email}")
                    return None
                
                user.last_login = datetime.utcnow()
                logger.info(f"User authenticated: {user.id} ({email})")
                return user
        
        logger.warning(f"Failed authentication attempt: {email}")
        return None
    
    def create_access_token(self, user: User) -> str:
        """Create JWT access token for user"""
        try:
            # Calculate expiration
            expires_delta = timedelta(hours=self.token_expiration_hours)
            expires = datetime.utcnow() + expires_delta
            
            # Create token payload
            payload = {
                "user_id": user.id,
                "email": user.email,
                "role": user.role,
                "permissions": user.permissions,
                "exp": expires,
                "iat": datetime.utcnow(),
                "type": "access"
            }
            
            # Generate token
            token = jwt.encode(
                payload,
                self.secret_key,
                algorithm=self.algorithm
            )
            
            # Store session
            self._active_sessions[user.id] = {
                "user": user,
                "token": token,
                "created_at": datetime.utcnow(),
                "expires_at": expires
            }
            
            logger.info(f"Access token created for user: {user.id}")
            return token
            
        except Exception as e:
            logger.error(f"Token creation error: {e}")
            raise
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode JWT token"""
        try:
            # Decode token
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            
            # Check if session exists
            user_id = payload.get("user_id")
            if user_id not in self._active_sessions:
                logger.warning(f"Token for non-existent session: {user_id}")
                return None
            
            # Verify token matches session
            session = self._active_sessions[user_id]
            if session["token"] != token:
                logger.warning(f"Token mismatch for user: {user_id}")
                return None
            
            logger.info(f"Token verified for user: {user_id}")
            return payload
            
        except jwt.ExpiredSignatureError:
            logger.warning("Expired token provided")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None
        except Exception as e:
            logger.error(f"Token verification error: {e}")
            return None
    
    def get_user_by_token(self, token: str) -> Optional[User]:
        """Get user from valid token"""
        payload = self.verify_token(token)
        if not payload:
            return None
        
        user_id = payload.get("user_id")
        if user_id in self._users_cache:
            return self._users_cache[user_id]
        
        return None
    
    def revoke_token(self, token: str) -> bool:
        """Revoke JWT token"""
        payload = self.verify_token(token)
        if not payload:
            return False
        
        user_id = payload.get("user_id")
        if user_id in self._active_sessions:
            del self._active_sessions[user_id]
            logger.info(f"Token revoked for user: {user_id}")
            return True
        
        return False
    
    def revoke_user_sessions(self, user_id: str) -> int:
        """Revoke all sessions for a user"""
        if user_id in self._active_sessions:
            del self._active_sessions[user_id]
            logger.info(f"All sessions revoked for user: {user_id}")
            return 1
        
        return 0
    
    def get_active_sessions(self) -> List[Dict[str, Any]]:
        """Get all active user sessions"""
        sessions = []
        for user_id, session in self._active_sessions.items():
            sessions.append({
                "user_id": user_id,
                "user_email": session["user"].email,
                "created_at": session["created_at"].isoformat(),
                "expires_at": session["expires_at"].isoformat()
            })
        
        return sessions
    
    def cleanup_expired_sessions(self) -> int:
        """Remove expired sessions"""
        now = datetime.utcnow()
        expired_sessions = []
        
        for user_id, session in self._active_sessions.items():
            if session["expires_at"] <= now:
                expired_sessions.append(user_id)
        
        for user_id in expired_sessions:
            del self._active_sessions[user_id]
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
        
        return len(expired_sessions)
    
    def hash_password(self, password: str) -> str:
        """Hash a password"""
        return bcrypt.hashpw(
            password.encode('utf-8'),
            bcrypt.gensalt()
        ).decode('utf-8')
    
    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        return self._users_cache.get(user_id)
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        for user in self._users_cache.values():
            if user.email.lower() == email.lower():
                return user
        return None
    
    def update_user_permissions(self, user_id: str, permissions: List[str]) -> bool:
        """Update user permissions"""
        if user_id in self._users_cache:
            self._users_cache[user_id].permissions = permissions
            logger.info(f"Permissions updated for user: {user_id}")
            return True
        return False
    
    def deactivate_user(self, user_id: str) -> bool:
        """Deactivate user account"""
        if user_id in self._users_cache:
            self._users_cache[user_id].is_active = False
            self.revoke_user_sessions(user_id)
            logger.info(f"User deactivated: {user_id}")
            return True
        return False
    
    def activate_user(self, user_id: str) -> bool:
        """Activate user account"""
        if user_id in self._users_cache:
            self._users_cache[user_id].is_active = True
            logger.info(f"User activated: {user_id}")
            return True
        return False
    
    def get_all_users(self) -> List[User]:
        """Get all users"""
        return list(self._users_cache.values())
    
    def get_users_by_role(self, role: str) -> List[User]:
        """Get users by role"""
        return [user for user in self._users_cache.values() if user.role == role]
    
    def get_session_count(self) -> int:
        """Get number of active sessions"""
        return len(self._active_sessions)
    
    def get_user_count(self) -> int:
        """Get total number of users"""
        return len(self._users_cache)
