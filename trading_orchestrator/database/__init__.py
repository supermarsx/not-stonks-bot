# Database Package Initialization

import logging
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, AsyncEngine
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manages database connections and sessions for the trading orchestrator.
    """
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.engine: Optional[AsyncEngine] = None
        self.session_factory: Optional[sessionmaker] = None
        
    async def initialize(self):
        """
        Initialize database engine and session factory.
        """
        try:
            self.engine = create_async_engine(
                self.database_url,
                echo=False,
                pool_pre_ping=True,
                pool_recycle=3600
            )
            
            self.session_factory = sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            logger.info("Database engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database engine: {e}")
            raise
    
    async def get_session(self) -> AsyncSession:
        """
        Get a new database session.
        """
        if not self.session_factory:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        
        return self.session_factory()
    
    async def close(self):
        """
        Close database connections.
        """
        if self.engine:
            await self.engine.dispose()
            logger.info("Database connections closed")


# Global database manager instance
db_manager: Optional[DatabaseManager] = None


def get_db_manager() -> Optional[DatabaseManager]:
    """
    Get the global database manager instance.
    """
    return db_manager


async def initialize_database(database_url: str) -> DatabaseManager:
    """
    Initialize the global database manager.
    """
    global db_manager
    
    db_manager = DatabaseManager(database_url)
    await db_manager.initialize()
    
    return db_manager


async def get_database_session() -> AsyncSession:
    """
    Get a database session from the global manager.
    """
    if not db_manager:
        raise RuntimeError("Database not initialized. Call initialize_database() first.")
    
    return await db_manager.get_session()


async def close_database():
    """
    Close the global database connection.
    """
    global db_manager
    
    if db_manager:
        await db_manager.close()
        db_manager = None