from pydantic import BaseModel, Field
from typing import Optional

class ThoughtResponseSchema(BaseModel):
    thought: str = Field(..., description="A description of the thought process leading to the response.")
    route: Optional[str] = Field(None, description="An optional field indicating the route or action taken, if any.")
    response: str = Field(..., description="The response message provided to the user.")
