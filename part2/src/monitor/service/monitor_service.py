from fastapi import APIRouter, status, Query, Path, Request
import uuid
from src.schemas import monitor_schemas
from pathlib import Path
from datetime import datetime as dt
from src.exceptions.custom_exception import (
    APIAuthenticationFailException, 
    InternalServerException, 
    RecordNotFoundException, 
    UnsupportedFileFormatException,
    ChatNotFoundException,
)
from src.config.pgvector_policy_client import PostgresVectorClient
from src.utils.database_manager import db_manager
from src.utils.app_notification_message import NotificationMessage
from src.utils.app_utils import  AppUtil
from src.config.settings import settings
from src.config.logger import Logger
from typing import List, Dict, Union
import json

logger = Logger(__name__)

class MonitorService:
    def __init__(self):
        self.vector_service = PostgresVectorClient()

    async def health_check(self):
        try:
            # Check database connection
            db_connected = False
            try:
                async with db_manager.get_connection() as conn:
                    await conn.fetchval("SELECT 1")
                    db_connected = True
            except Exception:
                pass
            
            # Check embedding model
            model_loaded = hasattr(self.vector_service, 'model') and self.vector_service.model is not None
            status = "healthy" if db_connected and model_loaded else "degraded"
            logger.info("Request processed successfully")

            return monitor_schemas.HealthResponse(
                status=status,
                database_connected=db_connected,
                embedding_model_loaded=model_loaded,
                timestamp=dt.now()
            ).model_dump()
        
        except Exception as ex:
            logger.error(f"Processing Request -> API v1/monitor/health/: {ex}")
            return monitor_schemas.HealthResponse(
                status="unhealthy",
                database_connected=False,
                embedding_model_loaded=False,
                timestamp=dt.now()
            ).model_dump()
    
    async def get_metrics(self):
        try:
            db_stats = await self.vector_service.get_database_stats()

            return monitor_schemas.MetricsResponse(
                total_policies=db_stats['total_policies'],
                total_embeddings=db_stats['total_policies'],
                database_size_mb=db_stats.get('database_size', '0MB').replace('MB', '')
            ).model_dump()
        except Exception as ex:
            logger.error(f"Processing Request -> API v1/monitor/metrics/: {ex}")
            raise InternalServerException()
    
    async def get_data_summary(self):
        try:
            db_stats = await self.vector_service.get_database_stats()
            return db_stats
        except Exception as ex:
            logger.error(f"Processing Request -> API v1/monitor/data-summary/: {ex}")
            raise InternalServerException()