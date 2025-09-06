from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.utils.connection import app_lifespan
from src.exceptions.custom_exception import (
    RecordNotFoundException, InternalServerException, AuthenticationFailException,
    UnsupportedFileFormatException, APIAuthenticationFailException
)
from src.exceptions.custom_exception_handler import ExceptionHandlers, AppException
from src.chat.controller import chat
from src.monitor.controller import monitor


app = FastAPI(
    title="Insurance RAG API", version="0.0.1", root_path="/v1", 
    description="Production-ready Agentic RAG system for insurance policy management",
    lifespan=app_lifespan
)

allow = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow,
    allow_credentials=True,
    allow_methods=allow,
    allow_headers=allow,
)

app.add_exception_handler(APIAuthenticationFailException, ExceptionHandlers.handle_authentication_fail_exception)
app.add_exception_handler(RecordNotFoundException, ExceptionHandlers.handle_record_not_found_exception)
app.add_exception_handler(AuthenticationFailException, ExceptionHandlers.handle_authentication_fail_exception)
app.add_exception_handler(InternalServerException, ExceptionHandlers.handle_internal_server_exception)
app.add_exception_handler(UnsupportedFileFormatException, ExceptionHandlers.handle_unsupported_file_format_exception)
app.add_exception_handler(Exception, ExceptionHandlers.handle_generic_exception)
app.add_exception_handler(AppException, ExceptionHandlers.handle_app_exception)

@app.get("/")
def home():
    return {"message": "Welcome to Insurance RAG API!"}

app.include_router(chat.chat_router)
app.include_router(monitor.monitor_router)