"""
Database Migration Manager
Handles database schema migrations and versioning
"""

import os
import asyncio
from pathlib import Path
from typing import List, Optional
from sqlalchemy import text, MetaData, Table
from sqlalchemy.ext.asyncio import AsyncConnection
from config.database import engine

class MigrationManager:
    """Manages database migrations and schema versioning"""
    
    def __init__(self):
        self.migrations_dir = Path("database/migrations")
        self.migrations_dir.mkdir(exist_ok=True)
        
    async def run_migrations(self):
        """Run all pending migrations"""
        try:
            # Create migrations table if it doesn't exist
            await self._create_migration_table()
            
            # Get current version
            current_version = await self._get_current_version()
            
            # Get pending migrations
            pending_migrations = self._get_pending_migrations(current_version)
            
            # Run pending migrations
            for migration in pending_migrations:
                await self._run_migration(migration)
                
        except Exception as e:
            print(f"❌ Migration failed: {str(e)}")
            raise
    
    async def _create_migration_table(self):
        """Create migrations tracking table"""
        async with engine.connect() as conn:
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    description TEXT,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            await conn.commit()
    
    async def _get_current_version(self) -> Optional[int]:
        """Get current database version"""
        async with engine.connect() as conn:
            try:
                result = await conn.execute(text("""
                    SELECT MAX(version) as current_version 
                    FROM schema_migrations
                """))
                row = result.fetchone()
                return row[0] if row and row[0] else 0
            except:
                return 0
    
    def _get_pending_migrations(self, current_version: int) -> List[int]:
        """Get list of pending migration versions"""
        migration_files = list(self.migrations_dir.glob("*.sql"))
        migration_versions = []
        
        for file in migration_files:
            try:
                version = int(file.stem.split('_')[0])
                if version > current_version:
                    migration_versions.append(version)
            except (ValueError, IndexError):
                continue
        
        return sorted(migration_versions)
    
    async def _run_migration(self, version: int):
        """Run a single migration"""
        migration_file = self.migrations_dir / f"{version:03d}_*.sql"
        migration_files = list(self.migrations_dir.glob(f"{version:03d}_*.sql"))
        
        if not migration_files:
            print(f"⚠️ Migration file not found for version {version}")
            return
        
        migration_path = migration_files[0]
        description = migration_path.stem.split('_', 1)[1] if '_' in migration_path.stem else ""
        
        try:
            # Read migration SQL
            with open(migration_path, 'r') as f:
                migration_sql = f.read()
            
            # Execute migration
            async with engine.connect() as conn:
                async with conn.begin():
                    # Split by semicolon and execute each statement
                    statements = [s.strip() for s in migration_sql.split(';') if s.strip()]
                    for statement in statements:
                        if statement:
                            await conn.execute(text(statement))
                    
                    # Record migration
                    await conn.execute(text("""
                        INSERT INTO schema_migrations (version, description)
                        VALUES (:version, :description)
                    """), {"version": version, "description": description})
                    
            print(f"✅ Migration {version} applied: {description}")
            
        except Exception as e:
            print(f"❌ Migration {version} failed: {str(e)}")
            raise
    
    async def create_migration(self, description: str) -> str:
        """Create a new migration file"""
        next_version = await self._get_next_version()
        filename = f"{next_version:03d}_{description.lower().replace(' ', '_')}.sql"
        filepath = self.migrations_dir / filename
        
        template = f"""-- Migration: {description}
-- Version: {next_version}
-- Created: {asyncio.get_event_loop().time()}

-- Add your SQL statements here
"""
        
        with open(filepath, 'w') as f:
            f.write(template)
        
        print(f"✅ Migration file created: {filepath}")
        return str(filepath)
    
    async def _get_next_version(self) -> int:
        """Get next migration version number"""
        current_version = await self._get_current_version()
        return (current_version or 0) + 1
    
    async def rollback_migration(self, version: int):
        """Rollback a specific migration"""
        print(f"⚠️ Rollback functionality not implemented yet for version {version}")
        # TODO: Implement rollback functionality
    
    async def get_migration_status(self) -> dict:
        """Get status of all migrations"""
        async with engine.connect() as conn:
            result = await conn.execute(text("""
                SELECT version, description, applied_at 
                FROM schema_migrations 
                ORDER BY version
            """))
            applied = result.fetchall()
            
            all_migrations = list(self.migrations_dir.glob("*.sql"))
            pending = self._get_pending_migrations(await self._get_current_version())
            
            return {
                "applied_migrations": [
                    {"version": v, "description": d, "applied_at": t} 
                    for v, d, t in applied
                ],
                "pending_migrations": pending,
                "total_applied": len(applied),
                "total_pending": len(pending)
            }