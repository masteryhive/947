from fastapi import FastAPI
from src.monitor.service.monitor_service import MonitorService
from src.chat.service.chat_service import ChatService
from src.utils.database_manager import db_manager
from contextlib import asynccontextmanager
from src.config.config_helper import Configuration
from src.config.settings import settings
from src.config.pgvector_policy_client import PostgresVectorClient
from src.config.logger import Logger
from typing import Dict
import os

logger = Logger(__name__)

@asynccontextmanager
async def app_lifespan(app: FastAPI):
    logger.info("Starting Insurance RAG API...")

    try:
        await db_manager.create_pool()
        vector_service = PostgresVectorClient()
        await vector_service.create_tables()

        # Initialize services
        # rag_service = AgenticRAGService()
        monitor_service = MonitorService()
        chat_service = ChatService()
        
        # Store services in the app state
        app.state.monitor_service = monitor_service
        app.state.chat_service = chat_service
        logger.info("All services initialized successfully")
        
        yield
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise

    finally:
        logger.info("Shutting down Insurance RAG API...")
        await db_manager.close_pool()