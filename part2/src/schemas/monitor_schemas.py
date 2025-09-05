from pydantic import BaseModel
from datetime import datetime


class HealthResponse(BaseModel):
    status: str
    database_connected: bool
    embedding_model_loaded: bool
    timestamp: datetime
    version: str = "1.0.0"

class MetricsResponse(BaseModel):
    total_policies: int
    total_embeddings: int
    database_size_mb: float