from fastapi import FastAPI
from monitor.service.monitor_service import MonitorService
from src.utils.database_manager import db_manager
# from motor.motor_asyncio import AsyncIOMotorClient
# from src.chat.model import chat
# from src.client.atlas_client import AtlasClient
from contextlib import asynccontextmanager
from src.config.config_helper import Configuration
from src.config.settings import settings
from src.config.logger import Logger
from typing import Dict
import os

logger = Logger(__name__)

# Global services
rag_service = None
monitor_service = None
vector_service = None

@asynccontextmanager
async def app_lifespan(app: FastAPI):

    logger.info("Starting Insurance RAG API...")
    global rag_service, ingestion_service, vector_service

    try:
        await db_manager.create_pool()
        # vector_service = PostgresVectorClient()
        # await vector_service.create_tables()

        # Initialize services    
        # rag_service = AgenticRAGService()
        monitor_service = MonitorService()
        logger.info("All services initialized successfully")
        
        yield
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise

    finally:
        logger.info("Shutting down Insurance RAG API...")
        await db_manager.close_pool()