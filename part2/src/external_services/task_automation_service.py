from .http_config import ApiClient
from src.chat.schemas import chat_schemas
from src.config.settings import settings
from typing import Optional, Literal
from datetime import datetime as dt
from datetime import time
from .base import BaseService
from src.utils.app_error_messages import ErrorMessages
from src.exceptions.custom_exception_handler import AppException
from ..config.logger import Logger
logger = Logger(__name__)

class TaskAutomationService(BaseService):
    def __init__(self):
        self.api = ApiClient(settings.MAP_BASE_URL)

    async def automate(
        self,
        title: str,
        description: str,
        date: dt,
        startTime: time,
        endTime: time,
        notify: time,
        task_type: Literal["EVENT", "APPOINTMENT", "SHIFT", "TASK"] = "EVENT",
        category: str = "client",
        merchant_id: Optional[str] = None,
        location_id: Optional[str] = None,
        department: Optional[str] = "",
        link: Optional[str] = "",
        staff: Optional[str] = "",
        priority: Optional[str] = "LOW",
        appointment_type: Optional[str] = None,
        shift_type: Optional[str] = None,
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
        # 1) prepare payload & endpoint
        create_endpoint = self._build_create_path(
            category,
            merchant_id=merchant_id,
            location_id=location_id,
            ticket_id=None,
            task="schedule",
        )

        # 2) build the payload object (no API call yet)
        if task_type == chat_schemas.EventType.EVENT:
            logger.info(f"date={date}, startTime={startTime}, endTime={endTime}, notify={notify}")
            payload = chat_schemas.SchedulePayload(
                title=title,
                description=description,
                type=chat_schemas.EventType.EVENT,
                merchantId=merchant_id,
                locationId=location_id,
                date=date,
                startTime=startTime,
                endTime=endTime,
                notify=notify,
                department=department,
                link=link,
                staff=staff,
            )
            operation = "Reminder Created for Event"
            user_message = (
                f"I've created the reminder for date {date}, which starts at {startTime} "
                f"and ends at {endTime}."
            )
        elif task_type == chat_schemas.EventType.TASK:
            payload = chat_schemas.SchedulePayload(
                title=title,
                description=description,
                type=chat_schemas.EventType.TASK,
                merchantId=merchant_id,
                locationId=location_id,
                date=date,
                startTime=startTime,
                endTime=endTime,
                notify=notify,
                department=department,
                priorityLevel=priority,
                staff=staff,
            )
            operation = "Task Created"
            user_message = (
                f"I've created the task for date {date}, which starts at {startTime}, "
                f"ends at {endTime}, description “{description}” and priority “{priority}.”"
            )
        elif task_type == chat_schemas.EventType.APPOINTMENT:
            payload = chat_schemas.SchedulePayload(
                title=title,
                description=description,
                type=chat_schemas.EventType.APPOINTMENT,
                merchantId=merchant_id,
                locationId=location_id,
                date=date,
                startTime=startTime,
                endTime=endTime,
                notify=notify,
                department=department,
                appointmentType=appointment_type,
                staff=staff,
            )
            operation = "Appointment Created"
            user_message = (
                f"I've created the appointment for date {date}, which starts at {startTime}, "
                f"ends at {endTime}, description “{description}” and appointment type “{appointment_type}.”"
            )
        else:  # SHIFT
            payload = chat_schemas.SchedulePayload(
                title=title,
                description=description,
                type=chat_schemas.EventType.SHIFT,
                merchantId=merchant_id,
                locationId=location_id,
                date=date,
                startTime=startTime,
                endTime=endTime,
                notify=notify,
                department=department,
                shiftType=shift_type,
                staff=staff,
            )
            operation = "Shift Created"
            user_message = (
                f"I've created the shift for date {date}, which starts at {startTime}, "
                f"ends at {endTime}, description “{description}” and shift type “{shift_type}.”"
            )
        
        logger.info(payload.model_dump())
        raw = payload.model_dump()
        payload_dict = { k: v for k, v in raw.items() if v is not None }

        # 3) actually make the HTTP call
        async with self.api as api:
            try:
                result = await api.post(
                    create_endpoint,
                    headers=headers,
                    payload=payload_dict,
                )
                result.raise_for_status()
            except Exception as ex:
                error = ErrorMessages.HANDLE_ERROR.format(error=ex)
                logger.error(error)
                raise AppException(error)

        # 4) now build and return your chat_schemas.Result
        return chat_schemas.Result(
            content=user_message,
            success=True,
            operation=operation,
            response=result.json(),
        )