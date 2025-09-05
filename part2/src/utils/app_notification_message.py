from pydantic import BaseModel


class NotificationMessage:
    FAQ_ADDED_SUCCESSFULLY = "FAQs added successfully"
    RETRIEVED_SUCCESSFULLY = "Retrieved successfully"
    INTENT_DELETED_SUCCESSFULLY = "Intent deleted successfully"
    INTERNAL_SERVER_ERROR = "Could not complete the request at the moment"
    RECORD_NOT_FOUND = "Record with specified constraints not found"