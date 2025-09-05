from fastapi import APIRouter, status, Query, Request, UploadFile, File, Depends, Path
from src.chat.service.chat_service import ChatService
from src.schemas import monitor_schemas, chat_schemas
from src.utils.app_utils import AppUtil
from typing import Dict, Union, List, Annotated
from src.utils.request_response import ApiResponse
import uuid

chat_router = APIRouter(prefix="/chats", tags=["Chat Session"])

@chat_router.post(
    '/ask-agent/{user_id}',
    response_model=Dict[str, Union[str, List]],
    summary="Ask MAP",
    status_code=status.HTTP_200_OK
)
async def ask_agent(
    chat: chat_schemas.ChatIn,
    user_id: Annotated[Union[uuid.UUID, str], Path(..., title="User ID to query")],
    chat_service: ChatService = Depends(AppUtil.get_chat_service)
):
    result = await chat_service.query(chat, user_id)
    return ApiResponse(
        message="Chat processed successfully",
        data=result
    )

@chat_router.post(
    '/ingest-excel/{user_id}',
    response_model=Dict[str, Union[str, chat_schemas.IngestionResponse]],
    summary="Insert a Excel document into a specified collection",
    status_code=status.HTTP_200_OK
)
async def ingest_excel(
    request: Request,
    user_id: Annotated[Union[uuid.UUID, str], Path(..., title="User ID to store")],
    file: UploadFile = File(..., title="Document file to upload"),
    chunk_size: Annotated[int, Query(title="Chunk size")] = 1000,
    chunk_overlap: Annotated[int, Query(title="Chunk overlap")] = 200,
    chat_service: ChatService = Depends(AppUtil.get_chat_service)
):
    result = await chat_service.ingest_excel(file, user_id, chunk_size, chunk_overlap)
    message = result.pop('message')
    return ApiResponse(
        message=message,
        data=result
    )