from .http_config import ApiClient
from src.chat.schemas import chat_schemas
from src.config.settings import settings
from src.utils.app_error_messages import ErrorMessages
from src.exceptions.custom_exception_handler import AppException
from .base import BaseService
from typing import Optional
from ..config.logger import Logger
logger = Logger(__name__)

class EscalationService(BaseService):
    def __init__(self):
        self.api = ApiClient(settings.MAP_BASE_URL)

    async def escalate(
        self,
        issue_description: str,
        subject: str,
        priority: str,
        category: str = "client",
        merchant_id: Optional[str] = None,
        location_id: Optional[str] = None,
        bearer_token: Optional[str] = None,
    ) -> chat_schemas.Result:
        
        if not bearer_token:
            raise AppException(ErrorMessages.AUTHENTICATION_REQUIRED)
        if not category:
            raise AppException(ErrorMessages.CATEGORY_REQUIRED)

        headers = {
            "Authorization": f"Bearer {bearer_token}",
            "Content-Type": "application/json",
        }
        async with self.api as api:
            # 1) generate ticket ID
            resp = await api.get(settings.TICKET_ID_GENENRATION_URL, headers=headers)
            resp.raise_for_status()
            gen = chat_schemas.GenerateTicketResponse(**resp.json().get("data", {}))
            ticket_id = gen.ticketId or (_ for _ in ()).throw(AppException(ErrorMessages.TOKEN_ID_REQUIRED))

            # 2) prepare payload & endpoint
            create_endpoint = self._build_create_path(category, merchant_id, location_id, ticket_id, task="escalation")
            payload = chat_schemas.CreateTicketPayload(
                message=issue_description,
                subject=subject,
                priority=priority,
                attachments=[],
            )

            # 3) post the ticket
            try:
                result = await api.post(create_endpoint, headers=headers, payload=payload.model_dump())
                result.raise_for_status()
            except Exception as ex:
                error = ErrorMessages.HANDLE_ERROR.format(error=ex)
                logger.error(error)
                raise AppException(error)

        logger.info(f"Escalation created: ticket={ticket_id}")
        return chat_schemas.Result(
            content=(
                "I've gathered all the necessary information and escalated this to our support team. "
                "They will contact you shortly to resolve the issue."
            ),
            success=True,
            operation="Escalation filled",
            ticketId=ticket_id,
            response=result.json(),
        )