from pydantic import BaseModel, Field, validator
from typing import List, Dict
from datetime import datetime as dt
from uuid import UUID, uuid4
import uuid
from src.schemas import monitor_schemas, chat_schemas, search_schema
from typing import Optional, Union, Any, List, Literal, Dict
from datetime import datetime as dt, timezone, time
from enum import Enum
import json


#-------------------------------------------------------chatsessionschema---------------------------------------------------------
class BaseSchema(BaseModel):
    class Config:
        from_attributes = True

        @staticmethod
        def json_schema_extra(schema: dict, _):
            props = {k: v for k, v in schema.get('properties', {}).items() if not v.get("hidden", False)}
            schema["properties"] = props

class InsurancePolicySchema(BaseModel):
    policy_number: str = Field(..., description="Unique policy identifier")
    insured_name: str = Field(..., description="Name of the insured party")
    sum_insured: float = Field(..., gt=0, description="Total sum insured")
    premium: float = Field(..., gt=0, description="Premium amount")
    own_retention_ppn: Optional[float] = Field(None, description="Own retention percentage")
    own_retention_sum_insured: Optional[float] = Field(None, description="Own retention sum insured")
    own_retention_premium: Optional[float] = Field(None, description="Own retention premium")
    treaty_ppn: Optional[float] = Field(None, description="Treaty percentage")
    treaty_sum_insured: Optional[float] = Field(None, description="Treaty sum insured")
    treaty_premium: Optional[float] = Field(None, description="Treaty premium")
    insurance_period_start_date: dt = Field(..., description="Insurance period start date")
    insurance_period_end_date: dt = Field(..., description="Insurance period end date")
    user_id: str = Field(..., description="User Id")
    
    @validator('insurance_period_end_date')
    def validate_end_date(cls, v, values):
        if 'insurance_period_start_date' in values and v <= values['insurance_period_start_date']:
            raise ValueError('End date must be after start date')
        return v

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Natural language question")
    user_id: Optional[str] = Field(None, description="User identifier for personalized results")
    limit: Optional[int] = Field(10, ge=1, le=50, description="Maximum number of results")
    include_metadata: Optional[bool] = Field(True, description="Include metadata in response")

class QueryResponse(BaseModel):
    answer: str = Field(..., description="Generated answer")
    sources: List[Dict[str, Any]] = Field(..., description="Source documents used")
    confidence_score: Optional[float] = Field(None, description="Confidence in the answer")
    processing_time: float = Field(..., description="Time taken to process query")
    query_id: str = Field(..., description="Unique query identifier")

class IngestionRequest(BaseModel):
    file_name: str
    chunk_size: Optional[int] = Field(1000, ge=100, le=5000)
    chunk_overlap: Optional[int] = Field(200, ge=0, le=1000)

class IngestionResponse(BaseModel):
    message: str
    records_processed: int
    records_inserted: int
    errors: List[str]
    processing_time: float
    ingestion_id: str

class RetreiveSchema(BaseSchema):
    user_id: str
    session_id: Optional[Union[UUID, str]] = Field(default=None, hidden=True)

class ChatIn(BaseModel):
    question: str

class UpdateSessionRequest(BaseModel):
    title: str

class FilePayload(BaseModel):
    filename: str
    content_type: str
    b64: str

class Content(BaseModel):
    session_id: Optional[str] = Field(
        None,
        description="Optional session UUID",
        example="ffe77d32-449b-47fa-8776-b01eb005cb51"
    )
    message: Optional[str] = Field(
        None,
        description="The user’s chat message",
        example="I'm unable to access patient records…"
    )
    payload: List[Any] = Field(
        default_factory=list,
        description="Any additional structured data"
    )

class ChatRequest(BaseModel):
    collection: str = Field(
        ...,
        description="Name of the collection",
        example="client"
    )
    content: Content


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, dt):
            # ensure UTC and format with trailing Z
            utc = obj.astimezone(timezone.utc)
            # "%f" is microseconds; we slice to milliseconds
            ms = f"{utc.microsecond//1000:03d}"
            return utc.strftime(f"%Y-%m-%dT%H:%M:%S.{ms}Z")
        if isinstance(obj, time):
            return obj.isoformat()
        return super().default(obj)

