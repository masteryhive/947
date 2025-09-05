from fastapi import HTTPException, status

class RecordNotAllowedException(HTTPException):
    def __init__(self, detail="Record not allowed"):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail,
        )

class RecordNotFoundException(HTTPException):
    def __init__(self, detail="Record not found"):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=detail,
        )

class AuthenticationFailException(HTTPException):
    def __init__(self, detail="Invalid token/user does not exists"):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
        )

class APIAuthenticationFailException(HTTPException):
    def __init__(self, detail="Invalid api key/user does not exists"):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
        )

class ChatNotFoundException(HTTPException):
    def __init__(self, detail="Chat not provided"):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail,
        )

class UploadFailException(HTTPException):
    def __init__(self, detail="Invalid account does not exists"):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
        )

class InternalServerException(HTTPException):
    def __init__(self, detail="Internal server error"):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail,
        )

class MAPAgentException(HTTPException):
    def __init__(self, detail="MAPAgent not initialized"):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail,
        )

class UnsupportedFileFormatException(HTTPException):
    def __init__(self, detail="Unsupported file format"):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail,
        )

class AppException(Exception):
    """Base class for all application-specific exceptions."""
    def __init__(self, message: str, status_code: int = status.HTTP_400_BAD_REQUEST):
        super().__init__(message)
        self.status_code = status_code
        self.detail = message

class JSONParseException(AppException):
    """Exception raised when JSON parsing fails."""
    def __init__(self, detail="Not a valid JSON file format"):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail,
        )