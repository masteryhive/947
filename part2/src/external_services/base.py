from src.config.settings import settings
from src.utils.app_error_messages import ErrorMessages
from src.exceptions.custom_exception_handler import AppException
from typing import Optional

class BaseService:
    def _build_create_path(
        self,
        category: str,
        merchant_id: Optional[str],
        location_id: Optional[str],
        ticket_id: Optional[str] = None,
        task="escalation"
    ) -> str:
        if category == "client":
            if not (merchant_id and location_id):
                raise AppException(ErrorMessages.MERCHANT_REQUIRED)
            if ticket_id and task=="escalation":
                return f"{settings.SUPPORT_SERVICES_URL}/{category}/create-ticket/{merchant_id}/{location_id}/{ticket_id}"
            elif task=="schedule":
                return f"{settings.SCHEDULE_SERVICES_URL}/common/create"
        elif category == "supplier":
            if not merchant_id:
                raise AppException(ErrorMessages.MERCHANT_REQUIRED)
            if ticket_id and task=="escalation":
                return f"{settings.SUPPORT_SERVICES_URL}/{category}/create-ticket/{merchant_id}/{ticket_id}"
            elif task=="schedule":
                return f"{settings.SCHEDULE_SERVICES_URL}/common/create"
        else:
            raise AppException("Invalid category")