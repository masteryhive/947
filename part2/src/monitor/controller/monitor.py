from fastapi import APIRouter, status, Query, Path, Request, Depends
from src.schemas import monitor_schemas
from src.utils.app_utils import AppUtil
from src.monitor.service.monitor_service import MonitorService
from typing import Dict, Union, List, Annotated
from src.utils.request_response import ApiResponse
import uuid

monitor_router = APIRouter(prefix="/monitor", tags=["Monitoring"])

@monitor_router.get(
    '/health',
    response_model=Dict[str, Union[str, List]],
    status_code=status.HTTP_200_OK
)
async def health_check(
    request: Request,
    monitor_service: MonitorService = Depends(AppUtil.get_monitor_service)
):
    result = await monitor_service.health_check()
    return ApiResponse(
        message="Request processed successfully",
        data=result
    )

@monitor_router.get(
    '/metrics',
    response_model=Dict[str, Union[str, List]],
    status_code=status.HTTP_200_OK
)
async def get_metrics(
    request: Request,
    monitor_service: MonitorService = Depends(AppUtil.get_monitor_service)
):
    result = await monitor_service.get_metrics()
    return ApiResponse(
        message="Request processed successfully",
        data=result
    )

@monitor_router.get(
    '/data-summary',
    response_model=Dict[str, Union[str, List]],
    status_code=status.HTTP_200_OK
)
async def get_data_summary(
    request: Request,
    monitor_service: MonitorService = Depends(AppUtil.get_monitor_service)
):
    result = await monitor_service.get_data_summary()
    return ApiResponse(
        message="Request processed successfully",
        data=result
    )