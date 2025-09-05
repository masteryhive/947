import asyncpg
import os
from typing import Optional
from contextlib import asynccontextmanager
from src.config.settings import settings
from src.config.logger import Logger
from typing import Dict
import os

logger = Logger(__name__)

class DatabaseManager:
    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None
        self.db_config = {
            'host': settings.POSTGRES_HOST,
            'port': int(settings.POSTGRES_PORT),
            'database': settings.POSTGRES_DB,
            'user': settings.POSTGRES_USER,
            'password': settings.POSTGRES_PASSWORD,
        }
    
    async def create_pool(self):
        """Create database connection pool"""
        try:
            self.pool = await asyncpg.create_pool(
                **self.db_config,
                min_size=5,
                max_size=20,
                command_timeout=60,
                server_settings={
                    'application_name': 'insurance_rag_app',
                }
            )
            logger.info("Database connection pool created successfully")
            
            # Enable pgvector extension
            async with self.pool.acquire() as conn:
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                
        except Exception as e:
            logger.error(f"Failed to create database pool: {e}")
            raise
    
    async def close_pool(self):
        """Close database connection pool"""
        if self.pool:
            await self.pool.close()
            logger.info("Database connection pool closed")
    
    @asynccontextmanager
    async def get_connection(self):
        """Get a database connection from the pool"""
        if not self.pool:
            await self.create_pool()
        
        async with self.pool.acquire() as connection:
            yield connection


# Global database manager instance
db_manager = DatabaseManager()