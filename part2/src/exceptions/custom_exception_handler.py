from fastapi import Request
from fastapi.responses import JSONResponse
from src.exceptions.custom_exception import (
    RecordNotFoundException, InternalServerException, AuthenticationFailException,
    UnsupportedFileFormatException, AppException
)
from src.utils.app_notification_message import NotificationMessage
from src.utils.request_response import ApiResponse
import json

class ExceptionHandlers:
    @staticmethod
    async def handle_record_not_found_exception(request: Request, exc: RecordNotFoundException) -> ApiResponse:
        return ApiResponse(
            code=exc.status_code,
            message=exc.detail,
        )

    @staticmethod
    async def handle_authentication_fail_exception(request: Request, exc: AuthenticationFailException) -> ApiResponse:
        return ApiResponse(
            code=exc.status_code,
            message=exc.detail,
        )

    @staticmethod
    async def handle_internal_server_exception(request: Request, exc: InternalServerException) -> ApiResponse:
        return ApiResponse(
            code=exc.status_code,
            message=exc.detail,
        )

    @staticmethod
    async def handle_generic_exception(request: Request, exc: Exception) -> ApiResponse:
        return ApiResponse(
            code=500,
            message=NotificationMessage.INTERNAL_SERVER_ERROR,
        )

    @staticmethod
    async def handle_unsupported_file_format_exception(request: Request, exc: UnsupportedFileFormatException) -> ApiResponse:
        return ApiResponse(
            code=exc.status_code,
            message=exc.detail,
        )

    @staticmethod
    async def handle_app_exception(request: Request, exc: AppException):
        return JSONResponse(
            status_code=exc.status_code,
            content=json.dumps({
                "success": False,
                "status": exc.detail
            })
        )
