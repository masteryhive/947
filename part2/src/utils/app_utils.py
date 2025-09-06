import os
import uuid
import io
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.config.settings import settings
import json
from datetime import datetime
import numpy as np
from typing import Union
from fastapi import Request, UploadFile, File
from src.utils.app_error_messages import ErrorMessages
from src.schemas.search_schema import MatchAnyOrInterval
from src.exceptions.custom_exception_handler import AppException

import pathlib
from typing import List, Dict, Any
from src.config.logger import Logger

from src.config.config_helper import Configuration
logger = Logger(__name__)

class AppUtil:
    @staticmethod
    def serialize_dict(a) -> dict:
        return {**{i: str(a[i]) for i in a if i == '_id'}, **{i: a[i] for i in a if i != '_id'}}

    @staticmethod
    def serialize_list(entity) -> list:
        return [AppUtil.serialize_dict(a) for a in entity]
    
    @staticmethod
    def load_file(path: pathlib.Path):
        with open(path, "r") as f:
            return f.read()
        
    @staticmethod
    def format(content):
        # Remove triple quotes if present
        if content.startswith('"""') and content.endswith('"""'):
            content = content[3:-3]
        elif content.startswith('"') and content.endswith('"'):
            content = content[1:-1]
        # Replace literal "\n" with actual newline characters
        content=content.replace("\\n", "\n")
        content=content.replace("\n\n", "\n") #this is done to be properly formatted on the front end
        return content
        
    @staticmethod 
    def clean_prefix(text, prefix = "Bearer "):
        if text.startswith(prefix):
            return text[len(prefix):]
        return text
    
    @staticmethod
    def remove_duplicates_preserve_order(duplicate_list):
        return list(dict.fromkeys(duplicate_list))
    
    @staticmethod
    def _extract_chat_input(chat_input):
        session_id = chat_input.get('content', {}).get("session_id")
        content = chat_input['content']
        message = content.get("message")
        
        category = chat_input.pop('collection', None)
        return (
            session_id, message, category
        )
    
    @staticmethod
    def extract(chat_input_org, schema=None):
        """
        Extracts and processes payload content from chat input.
        Supports pdf, images (jpeg, png), csv, and xlsx files.
        """
        chat_input = chat_input_org.copy()
        payload = chat_input.get("content", {}).get("payload", None)
        if payload is not None:
            # Directly return the payload content without unnecessary loops
            return chat_input
        return None
    
    @staticmethod
    def convert_value(value):
        if isinstance(value, (datetime, pd.Timestamp)):
            return value.isoformat() if pd.notna(value) else None
        elif isinstance(value, (np.number, float, int)):
            return float(value) if not pd.isna(value) else None
        elif pd.isna(value):
            return None
        return value
    
    @staticmethod
    async def get_chat_service(request: Request):
        return request.app.state.chat_service
    
    @staticmethod
    async def get_monitor_service(request: Request):
        return request.app.state.monitor_service