from beanie import Document
from typing import List, Dict, Optional, Union
from uuid import UUID, uuid4
from datetime import datetime as dt
from pydantic import BaseModel, Field
from src.chat.schemas.chat_schemas import Message

class ChatSession(Document):
    class DocumentMeta:
        collection_name = "chat_sessions"
    
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    user_id: str
    title: str = "New Chat"
    date_created: dt = dt.now()
    last_modified: dt = dt.now()

class ChatHistory(Document):
    class DocumentMeta:
        collection_name = "chat_history"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    user_id: str
    session_id: UUID
    messages: List[Message] = []