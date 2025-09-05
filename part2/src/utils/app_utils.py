import dotenv
dotenv.load_dotenv(dotenv.find_dotenv(), override=True)
import os
import uuid
import io
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.config.settings import settings
import json
from typing import Union

from src.exceptions.custom_exception import (
    APIAuthenticationFailException, 
    InternalServerException, 
    RecordNotFoundException, 
    UnsupportedFileFormatException,
    ChatNotFoundException, 
    MAPAgentException
)
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
    def embed_document(file_bytes, file_extension, document_type, user_id=None, chunk_size=500, chunk_overlap=20):
        """
        This function processes the file based on its type (pdf, image, csv, xlsx, video).
        It returns the processed content that will be uploaded later.
        """
        from part2.src.config.pgvector_client import AtlasClient

        docs_to_add: List[Dict[str, Any]] = []
        
        try:
            monogo_atlas = AtlasClient(
                user_id=user_id,
                altas_database=settings.UNIQUE_USER_COLLECTION
            )

            if file_extension == "pdf":
                docs = AppUtil.load_pdf(file_bytes)
            elif file_extension in ["docx", "doc"]:
                docs = AppUtil.load_docx(file_bytes)
            elif file_extension in ["jpeg", "png"]:
                docs = AppUtil.read_image(file_bytes)
            # elif file_extension in ["csv", "xlsx"]:
            #     return AppUtil.load_pandas(file_bytes, file_extension)
            # elif file_extension == "json":
            #     return AppUtil.load_json(file_bytes)
            else:
                raise ValueError("Unsupported file extension")
            
            # 2) Chunk those paragraph‑Documents into ~500‑token pieces
            splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            chunks = splitter.split_documents(docs)

            # 4) Convert to simple dicts and embed into user‑specific collection
            for chunk in chunks:
                chunk.metadata.update({
                    "document_type": document_type,
                    "_id" : str(uuid.uuid4()),
                    "user_id": user_id,
                    "collection_name": monogo_atlas.collection_name
                })
                # preserve existing metadata (_id, document_type, page, etc.)
                docs_to_add.append({
                    "page_content": chunk.page_content,
                    "metadata": chunk.metadata.copy()
                })

            # 5) Upsert into MongoDB Atlas under the collection named after this user
            monogo_atlas.insert(docs_to_add, is_batch=True)
            return True
        
        except Exception as ex:
            logger.error(f"Error Embedding document -> {ex}")
            return False
        
    @staticmethod
    def load_pandas(file_bytes, file_type, flag=None):
        try:
            if file_type == "csv":
                return pd.read_csv(io.BytesIO(file_bytes))
            elif file_type == "xlsx":
                if flag is None:
                    # Load normally with ExcelFile if flag is None
                    return pd.ExcelFile(io.BytesIO(file_bytes))
            else:
                raise ValueError("Unsupported file type")
        except Exception as ex:
            logger.error(f"Failed to load file into pandas: {ex}")
            return None
        
    @staticmethod
    def search_vector_db(question: str, client, document_type: str = None, user_id: str = None, session_id: str = None):
        ALLOWED_DOCUMENT_TYPES = ["overview", "inventory", "faq"]
        filters = {}

        if not question:
            return AppException(ErrorMessages.QUESTION_REQUIRED)
        elif session_id is None and document_type not in ALLOWED_DOCUMENT_TYPES:
            return f"Invalid document type: {document_type}. Allowed types are: {ALLOWED_DOCUMENT_TYPES}"
        else:
            try:
                if user_id is not None:
                    filters.update({
                        'user_id': MatchAnyOrInterval(any=[str(user_id)])
                    })
                    filters.update({
                        "document_type": MatchAnyOrInterval(eq=document_type)
                    })
                
                # Call the search method with the filters
                search_result = client.search(query=question, filters=filters)

                return json.dumps(
                    {
                        "question": question,
                        "search_result": search_result,
                        "success": True,
                        "status": "Searched the vector DB"
                    }
                )
            except Exception as ex:
                error = ErrorMessages.QUERY_ERROR.format(error=ex)
                logger.error(error)
                return AppException(error)