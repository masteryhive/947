from fastapi import APIRouter, status, Query, Request, UploadFile, File
from src.chat.service.chat_service import ChatService
from src.schemas import monitor_schemas, chat_schemas
from typing import Dict, Union, List, Annotated
from src.utils.request_response import ApiResponse
import uuid

chat_router = APIRouter(prefix="/chats", tags=["Chat Session"])

@chat_router.post(
    '/ask-agent/',
    response_model=Dict[str, Union[str, List]],
    summary="Ask MAP",
    status_code=status.HTTP_200_OK
)
async def ask_agent(request: Request):
    result = await ChatService.ask(request)
    return ApiResponse(
            message="Chat processed successfully",
            data=result
        )

@chat_router.post(
    '/ingest-excel/',
    response_model=Dict[str, Union[str, chat_schemas.IngestionResponse]],
    summary="Insert a Excel document into a specified collection",
    status_code=status.HTTP_200_OK
)
async def ingest_excel(
    file: UploadFile = File(..., title="Document file to upload"),
    chunk_size: Annotated[int, Query(title="Chunk size")] = 1000,
    chunk_overlap: Annotated[int, Query(title="Chunk overlap")] = 200
):
    result = await ChatService.ingest_excel(file, chunk_size, chunk_overlap)
    return ApiResponse(
        message="Document inserted successfully",
        data=result
    )