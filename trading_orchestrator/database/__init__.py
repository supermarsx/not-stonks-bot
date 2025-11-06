"""
Database Package Initialization
Provides database initialization and session management functions
"""

from .models import *
from .migrations.migration_manager import MigrationManager
from config.database import init_db, get_db, get_db_context, AsyncSessionLocal

async def init_database(database_config):
    """
    Initialize database with configuration
    
    Args:
        database_config: Database configuration object with url and echo settings
    """
    try:
        # Run database migrations if needed
        migration_manager = MigrationManager()
        await migration_manager.run_migrations()
        
        # Initialize database tables
        await init_db()
        
        return True
    except Exception as e:
        print(f"❌ Database initialization failed: {str(e)}")
        raise

async def get_database_session():
    """
    Get database session for dependency injection
    
    Returns:
        Async database session
    """
    async for session in get_db():
        yield session

async def close_database_connections():
    """Close all database connections"""
    from config.database import engine
    await engine.dispose()

# Export commonly used database functions
__all__ = [
    "init_database",
    "get_database_session", 
    "close_database_connections",
    "MigrationManager",
    "init_db",
    "get_db",
    "get_db_context",
    "AsyncSessionLocal"
]